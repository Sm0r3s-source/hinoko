use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use url::Url;

use crate::model::clip;

#[derive(Debug)]
pub struct Page {
    pub url: String,
    pub title: String,
    pub excerpt: String,
}

pub fn host_allowed(url: &Url) -> bool {
    if url.scheme() != "https" {
        return false;
    }
    if !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    // Only real domain names. The url crate normalises odd IP spellings
    // (0x7f.1, 2130706433, [::1]) into Host::Ipv4 / Host::Ipv6, which we refuse.
    let Some(url::Host::Domain(host)) = url.host() else {
        return false;
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() || !host.contains('.') {
        return false;
    }
    !(host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".lan")
        || host.ends_with(".home.arpa"))
}

pub fn redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= 2 {
            return attempt.error(std::io::Error::other("too many redirects"));
        }
        if host_allowed(attempt.url()) {
            attempt.follow()
        } else {
            attempt.error(std::io::Error::other("blocked redirect"))
        }
    })
}

/// Resolves names itself and drops every non-public address, so the check
/// happens at connect time. A pre-check followed by a second lookup would
/// leave a DNS-rebinding gap.
pub struct PublicOnly;

impl reqwest::dns::Resolve for PublicOnly {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let found = tokio::net::lookup_host((host.as_str(), 443)).await?;
            let ok: Vec<SocketAddr> = found.filter(|a| !ip_blocked(a.ip())).collect();
            if ok.is_empty() {
                return Err("that name does not resolve to a public address".into());
            }
            let addrs: reqwest::dns::Addrs = Box::new(ok.into_iter());
            Ok(addrs)
        })
    }
}

/// Client for reading pages. Proxies are off unless `allow_proxy`: a proxy
/// resolves names itself, which would bypass `PublicOnly`.
pub fn client(allow_proxy: bool) -> reqwest::Result<reqwest::Client> {
    let mut b = reqwest::Client::builder()
        .redirect(redirect_policy())
        .dns_resolver(std::sync::Arc::new(PublicOnly))
        .timeout(Duration::from_secs(15))
        .user_agent("Hinoko/0.2");
    if !allow_proxy {
        b = b.no_proxy();
    }
    b.build()
}

pub fn ip_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => ipv4_blocked(v),
        IpAddr::V6(v) => ipv6_blocked(v),
    }
}

fn ipv4_blocked(v: Ipv4Addr) -> bool {
    let o = v.octets();
    v.is_loopback()
        || v.is_unspecified()
        || v.is_link_local()
        || v.is_multicast()
        || v.is_broadcast()
        || v.is_private()
        || o[0] == 0
        || o[0] >= 224
        || (o[0] == 100 && (o[1] & 0b1100_0000) == 0b0100_0000)
        || (o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2))
        || (o[0] == 192 && o[1] == 88 && o[2] == 99)
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
}

fn ipv6_blocked(v: Ipv6Addr) -> bool {
    // ::ffff:a.b.c.d and the deprecated ::a.b.c.d
    if let Some(v4) = v.to_ipv4() {
        return ipv4_blocked(v4);
    }
    let s = v.segments();
    // NAT64 64:ff9b::/96 carries an IPv4 address in the last 32 bits
    if s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0u16; 4] {
        return ipv4_blocked(Ipv4Addr::new((s[6] >> 8) as u8, s[6] as u8, (s[7] >> 8) as u8, s[7] as u8));
    }
    // 6to4 2002::/16 carries one in bits 16..48
    if s[0] == 0x2002 {
        return ipv4_blocked(Ipv4Addr::new((s[1] >> 8) as u8, s[1] as u8, (s[2] >> 8) as u8, s[2] as u8));
    }
    v.is_loopback()
        || v.is_unspecified()
        || v.is_multicast()
        || (s[0] & 0xffc0) == 0xfe80
        || (s[0] & 0xffc0) == 0xfec0
        || (s[0] & 0xfe00) == 0xfc00
        || (s[0] == 0x2001 && (s[1] == 0 || s[1] == 0x0db8))
        || (s[0] == 0x0100 && s[1..4] == [0u16; 3])
}

pub async fn read(http: &reqwest::Client, raw: &str) -> Result<Page, String> {
    read_limit(http, raw, 1200).await
}

/// Like [`read`], keeping up to `limit` characters of page text.
pub async fn read_limit(http: &reqwest::Client, raw: &str, limit: usize) -> Result<Page, String> {
    let url = Url::parse(raw.trim()).map_err(|_| "that is not a URL".to_string())?;
    if !host_allowed(&url) {
        return Err("only public https pages are readable".into());
    }
    let resp = http
        .get(url.clone())
        .timeout(Duration::from_secs(8))
        .send()
        .await
        .map_err(|_| "the page did not answer".to_string())?;
    if !resp.status().is_success() {
        return Err(format!("the page returned {}", resp.status()));
    }
    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !(ctype.is_empty() || ctype.contains("text") || ctype.contains("xml") || ctype.contains("json")) {
        return Err("that page is not text".into());
    }
    let mut resp = resp;
    let mut buf: Vec<u8> = Vec::new();
    while buf.len() < 120_000 {
        match resp.chunk().await {
            Ok(Some(c)) => {
                let room = 120_000 - buf.len();
                if c.len() > room {
                    buf.extend_from_slice(&c[..room]);
                    break;
                }
                buf.extend_from_slice(&c);
            }
            Ok(None) => break,
            Err(_) => {
                if buf.is_empty() {
                    return Err("the page stopped early".into());
                }
                break;
            }
        }
    }
    let text = String::from_utf8_lossy(&buf);
    let (title, excerpt) = html_to_text_limit(&text, limit);
    let title = if title.is_empty() {
        url.host_str().unwrap_or("Page").to_string()
    } else {
        title
    };
    Ok(Page {
        url: url.to_string(),
        title: clip(&title, 140),
        excerpt: if excerpt.is_empty() {
            "The page had no readable text.".into()
        } else {
            excerpt
        },
    })
}

// ---- web search ----

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Search the web. With `searx` set (a SearXNG base URL, which may be on your
/// own network) its JSON API is used; otherwise DuckDuckGo's HTML page is
/// scraped, which is best-effort and keyless.
pub async fn search(
    plain: &reqwest::Client,
    web: &reqwest::Client,
    searx: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<Hit>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("empty query".into());
    }
    let searx = searx.trim().trim_end_matches('/');
    let hits = if !searx.is_empty() {
        let res = plain
            .get(format!("{searx}/search"))
            .query(&[("q", query), ("format", "json")])
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|_| "the search server did not answer".to_string())?;
        if !res.status().is_success() {
            return Err(format!("the search server returned {} (is the JSON format enabled?)", res.status()));
        }
        let body = res.text().await.map_err(|_| "unreadable search answer".to_string())?;
        parse_searx(&body)?
    } else {
        let res = web
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query)])
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|_| "search did not answer".to_string())?;
        if !res.status().is_success() {
            return Err(format!("search returned {}", res.status()));
        }
        let body = res.text().await.map_err(|_| "unreadable search answer".to_string())?;
        parse_ddg(&body)
    };
    Ok(hits.into_iter().take(limit).collect())
}

pub fn parse_searx(body: &str) -> Result<Vec<Hit>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|_| "the search server did not return JSON".to_string())?;
    let mut out = Vec::new();
    for r in v["results"].as_array().into_iter().flatten() {
        let (Some(url), Some(title)) = (r["url"].as_str(), r["title"].as_str()) else { continue };
        out.push(Hit { title: clip(title.trim(), 140), url: url.to_string(), snippet: clip(r["content"].as_str().unwrap_or("").trim(), 300) });
    }
    Ok(out)
}

pub fn parse_ddg(html: &str) -> Vec<Hit> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find("class=\"result__a\"") {
        let after = &rest[i..];
        let Some(href_at) = after.find("href=\"") else { break };
        let href_rest = &after[href_at + 6..];
        let Some(href_end) = href_rest.find('"') else { break };
        let href = decode_entities(&href_rest[..href_end]);
        let Some(gt) = href_rest[href_end..].find('>') else { break };
        let title_rest = &href_rest[href_end + gt + 1..];
        let title_end = title_rest.find("</a>").unwrap_or(title_rest.len());
        let title = decode_entities(&strip_tags(&title_rest[..title_end])).trim().to_string();
        // the snippet link follows the title, before the next result
        let block_end = title_rest.find("class=\"result__a\"").unwrap_or(title_rest.len());
        let block = &title_rest[..block_end];
        let snippet = block
            .find("class=\"result__snippet\"")
            .and_then(|s| {
                let b = &block[s..];
                let gt = b.find('>')?;
                let end = b[gt..].find("</a>").map(|e| gt + e).unwrap_or(b.len());
                Some(decode_entities(&strip_tags(&b[gt + 1..end])).trim().to_string())
            })
            .unwrap_or_default();
        // results are wrapped in a redirect: //duckduckgo.com/l/?uddg=<encoded url>
        let url = Url::parse(&format!("https:{}", href.trim_start_matches("https:")))
            .ok()
            .and_then(|u| u.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.to_string()))
            .or_else(|| href.starts_with("http").then(|| href.clone()));
        if let Some(url) = url {
            if !title.is_empty() {
                out.push(Hit { title: clip(&title, 140), url, snippet: clip(&snippet, 300) });
            }
        }
        rest = &title_rest[block_end..];
    }
    out
}

pub fn html_to_text(raw: &str) -> (String, String) {
    html_to_text_limit(raw, 1200)
}

pub fn html_to_text_limit(raw: &str, limit: usize) -> (String, String) {
    let without = cut_blocks(raw, "script");
    let without = cut_blocks(&without, "style");
    let title = between(&without, "title").unwrap_or_default();
    let text = collapse(&strip_tags(&without));
    (
        decode_entities(title.trim()),
        clip(&decode_entities(&text), limit),
    )
}

fn between(raw: &str, tag: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    let open = format!("<{tag}");
    let start = lower.find(&open)?;
    let after = raw.get(start..)?;
    let gt = after.find('>')?;
    let rest = after.get(gt + 1..)?;
    let lower_rest = rest.to_ascii_lowercase();
    let end = lower_rest.find(&format!("</{tag}"))?;
    Some(rest.get(..end)?.to_string())
}

fn cut_blocks(raw: &str, tag: &str) -> String {
    // to_ascii_lowercase keeps byte offsets identical, so indices from `lower`
    // are valid in `raw`. Every index below sits on an ASCII byte (or just
    // after one), hence on a char boundary.
    let lower = raw.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}");
    let mut out = String::new();
    let mut i = 0;
    while let Some(rel) = lower[i..].find(&open) {
        let start = i + rel;
        let from = start + open.len();
        let boundary = matches!(lower.as_bytes().get(from), None | Some(b'>' | b'/' | b' ' | b'\t' | b'\n' | b'\r'));
        if !boundary {
            // something like <scripture>: not our tag
            out.push_str(&raw[i..from]);
            i = from;
            continue;
        }
        out.push_str(&raw[i..start]);
        let Some(rel_end) = lower[from..].find(&close) else {
            return out; // unterminated block: drop the rest
        };
        let after = from + rel_end + close.len();
        match lower[after..].find('>') {
            Some(gt) => i = after + gt + 1,
            None => return out,
        }
    }
    out.push_str(&raw[i..]);
    out
}

fn strip_tags(input: &str) -> String {
    let mut out = String::new();
    let mut chars = input.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if ch == '<' {
            let rest = input.get(i..).unwrap_or("");
            let name: String = rest.chars().skip(1).take(6).collect::<String>().to_ascii_lowercase();
            if name.starts_with("br")
                || name.starts_with("p")
                || name.starts_with("/p")
                || name.starts_with("div")
                || name.starts_with("li")
                || name.starts_with("h1")
                || name.starts_with("h2")
                || name.starts_with("h3")
                || name.starts_with("tr")
            {
                out.push('\n');
            }
            for (_, c) in chars.by_ref() {
                if c == '>' {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn decode_entities(s: &str) -> String {
    let pairs = [
        ("nbsp;", " "),
        ("lt;", "<"),
        ("gt;", ">"),
        ("quot;", "\""),
        ("#39;", "'"),
        ("apos;", "'"),
        ("amp;", "&"), // last, so &amp;lt; ends as the text &lt;
    ];
    let mut out = s.to_string();
    for (ent, val) in pairs {
        out = out.replace(&format!("&{ent}"), val);
    }
    out
}

fn collapse(s: &str) -> String {
    let mut out = String::new();
    let mut blank = 0;
    for line in s.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank += 1;
            if blank <= 1 && !out.is_empty() {
                out.push('\n');
            }
        } else {
            blank = 0;
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&line);
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_local_and_plain_http() {
        assert!(host_allowed(&Url::parse("https://example.com/a").unwrap()));
        assert!(!host_allowed(&Url::parse("http://example.com/a").unwrap()));
        assert!(!host_allowed(&Url::parse("https://127.0.0.1/").unwrap()));
        assert!(!host_allowed(&Url::parse("https://localhost/x").unwrap()));
        assert!(!host_allowed(&Url::parse("https://foo.internal/x").unwrap()));
        assert!(!host_allowed(&Url::parse("https://user:pass@example.com/").unwrap()));
        assert!(ip_blocked(IpAddr::V4(Ipv4Addr::new(10, 1, 1, 1))));
        assert!(ip_blocked(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))));
        assert!(!ip_blocked(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
    }

    #[test]
    fn pulls_title_and_skips_script() {
        let raw = format!(
            "<html><title>Desk {} Co</title><script>secret()</script><p>Hello there</p></html>",
            format!("&{}", "amp;")
        );
        let (title, text) = html_to_text(&raw);
        assert_eq!(title, "Desk & Co");
        assert!(text.contains("Hello there"), "{text}");
        assert!(!text.contains("secret"), "{text}");
    }

    #[test]
    fn parses_searx_json() {
        let body = r#"{"results":[{"title":"Rust","url":"https://www.rust-lang.org/","content":"A language"},{"title":"no url"}]}"#;
        let hits = parse_searx(body).unwrap();
        assert_eq!(hits, vec![Hit { title: "Rust".into(), url: "https://www.rust-lang.org/".into(), snippet: "A language".into() }]);
        assert!(parse_searx("<html>").is_err());
    }

    #[test]
    fn parses_duckduckgo_html() {
        let html = r#"<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">Example <b>A</b></a>
<a class="result__snippet" href="x">First &amp; best <b>page</b></a></div>
<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Example B</a></div>"#;
        let hits = parse_ddg(html);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://example.com/a?x=1");
        assert_eq!(hits[0].title, "Example A");
        assert_eq!(hits[0].snippet, "First & best page");
        assert_eq!(hits[1].url, "https://example.org/b");
        assert_eq!(hits[1].snippet, "");
        assert!(parse_ddg("nothing here").is_empty());
    }

    #[test]
    fn unterminated_close_tag_before_multibyte_text_does_not_panic() {
        // used to slice inside a UTF-8 character
        let (_, text) = html_to_text("<p>hi</p><script>x</script日本語 more");
        assert!(!text.contains('x'));
        let (_, text) = html_to_text("<style>a{}</style é<p>after</p>");
        let _ = text;
        let _ = html_to_text("<script>never closed 日本");
        let _ = html_to_text("<scr\u{e9}ipt></script");
        let _ = html_to_text("</script日");
    }

    #[test]
    fn similar_tag_names_are_kept() {
        let (_, text) = html_to_text("<scripture>keep me</scripture><script>drop()</script>");
        assert!(text.contains("keep me"), "{text}");
        assert!(!text.contains("drop"), "{text}");
    }

    #[test]
    fn scraps_never_panic() {
        let bits = ["<", ">", "</", "<script", "</script", "</style", "<style>", "é", "日", "&amp;", "<title>", "</title>", "x", " "];
        let mut seed = 0x2545_f491_u64;
        for _ in 0..4000 {
            let mut s = String::new();
            for _ in 0..(seed % 9 + 1) {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s.push_str(bits[(seed >> 33) as usize % bits.len()]);
            }
            let _ = html_to_text(&s);
        }
    }

    #[test]
    fn entities_decode_once() {
        assert_eq!(decode_entities("a &amp;lt; b"), "a &lt; b");
    }

    #[test]
    fn odd_ip_spellings_are_refused() {
        for u in ["https://0x7f.1/", "https://2130706433/", "https://[::1]/", "https://127.1/", "https://foo.lan/"] {
            assert!(!host_allowed(&Url::parse(u).unwrap()), "{u}");
        }
    }

    #[test]
    fn wider_blocklist() {
        let blocked = ["::1", "::ffff:127.0.0.1", "::127.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::", "fe80::1", "fec0::1", "fd00::1", "2001:db8::1", "2001::1"];
        for b in blocked {
            assert!(ip_blocked(b.parse().unwrap()), "{b}");
        }
        for b in ["192.0.2.1", "198.51.100.7", "203.0.113.9", "192.88.99.1", "240.0.0.1", "172.20.0.1", "100.64.0.1"] {
            assert!(ip_blocked(b.parse().unwrap()), "{b}");
        }
        for ok in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111", "93.184.216.34"] {
            assert!(!ip_blocked(ok.parse().unwrap()), "{ok}");
        }
    }
}
