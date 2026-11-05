//! Ask the same question to several models at once and watch the answers side by side.
//! Nothing is saved to a chat; it is a scratch comparison.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;

use crate::engine::Engine;
use crate::llm;

#[derive(Clone, Debug, Default)]
pub struct Lane {
    pub model: String,
    pub text: String,
    pub think: String,
    pub error: String,
    pub done: bool,
    /// Time to the first piece of text, and to the end, in milliseconds.
    pub first_ms: u64,
    pub total_ms: u64,
    /// Tokens written, and whether that is an estimate.
    pub tokens: u64,
    pub estimated: bool,
    pub counted: bool,
    pub think_chars: usize,
}

impl Lane {
    /// Tokens per second while the model was writing (not counting the wait for the first piece).
    pub fn tokens_per_sec(&self) -> f32 {
        let writing = self.total_ms.saturating_sub(self.first_ms);
        let secs = if writing >= 200 { writing } else { self.total_ms };
        if secs == 0 {
            return 0.0;
        }
        self.tokens as f32 / (secs as f32 / 1000.0)
    }
}

#[derive(Default)]
pub struct Run {
    pub prompt: String,
    pub lanes: Vec<Lane>,
}

#[derive(Clone)]
pub struct Compare {
    pub run: Arc<Mutex<Run>>,
    stop: Arc<AtomicBool>,
}

impl Compare {
    /// A finished comparison with the given sides, for screenshots and tests.
    pub fn sample(prompt: &str, lanes: Vec<Lane>) -> Compare {
        Compare { run: Arc::new(Mutex::new(Run { prompt: prompt.to_string(), lanes })), stop: Arc::new(AtomicBool::new(true)) }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
    pub fn finished(&self) -> bool {
        self.run.lock().map(|r| r.lanes.iter().all(|l| l.done)).unwrap_or(true)
    }
}

impl Engine {
    /// Start one lane per provider id (`None` = the default model). Returns at once.
    pub fn compare_start(&self, prompt: &str, choices: Vec<Option<String>>) -> Compare {
        let settings = self.settings();
        let run = Arc::new(Mutex::new(Run { prompt: prompt.to_string(), lanes: vec![Lane::default(); choices.len()] }));
        let stop = Arc::new(AtomicBool::new(false));
        for (i, choice) in choices.into_iter().enumerate() {
            let Some(provider) = settings.provider_for(choice.as_deref()).filter(|p| p.ready()).cloned() else {
                if let Ok(mut r) = run.lock() {
                    r.lanes[i].error = "no model is set up for this side".into();
                    r.lanes[i].done = true;
                }
                continue;
            };
            if let Ok(mut r) = run.lock() {
                r.lanes[i].model = if provider.model.is_empty() { provider.name.clone() } else { format!("{} · {}", provider.name, provider.model) };
            }
            let (me, run, stop, prompt) = (self.clone(), run.clone(), stop.clone(), prompt.to_string());
            let temperature = settings.temperature;
            self.rt.spawn(async move {
                let mut provider = provider;
                let started = Instant::now();
                if provider.id == crate::local::PROVIDER_ID && !me.local_info().running {
                    if let Err(e) = me.ensure_local(&provider).await {
                        if let Ok(mut r) = run.lock() {
                            r.lanes[i].error = format!("couldn't start the local model: {e}");
                            r.lanes[i].done = true;
                        }
                        (me.repaint)();
                        return;
                    }
                }
                if let Some(p) = me.settings().provider(&provider.id) {
                    provider = p.clone();
                }
                let messages = vec![json!({"role": "user", "content": prompt})];
                let mut on = |piece: llm::Piece| -> bool {
                    if let Ok(mut r) = run.lock() {
                        let lane = &mut r.lanes[i];
                        match piece {
                            llm::Piece::Text(t) => {
                                if lane.first_ms == 0 {
                                    lane.first_ms = started.elapsed().as_millis() as u64;
                                }
                                lane.text.push_str(t);
                            }
                            llm::Piece::Think(t) => {
                                if lane.think.len() < 20_000 {
                                    lane.think.push_str(t);
                                }
                                lane.think_chars += t.len();
                                if lane.first_ms == 0 {
                                    lane.first_ms = started.elapsed().as_millis() as u64;
                                }
                            }
                            llm::Piece::Usage(n, est) => {
                                lane.tokens = n;
                                lane.estimated = est;
                                lane.counted = true;
                            }
                        }
                        lane.total_ms = started.elapsed().as_millis() as u64;
                        if !lane.counted {
                            lane.tokens = llm::estimate_tokens(lane.text.len() + lane.think_chars);
                            lane.estimated = true;
                        }
                    }
                    (me.repaint)();
                    !stop.load(Ordering::Relaxed)
                };
                let res = llm::stream(&me.http, &provider, temperature, &messages, 4096, None, &mut on).await;
                if let Ok(mut r) = run.lock() {
                    let lane = &mut r.lanes[i];
                    lane.total_ms = started.elapsed().as_millis() as u64;
                    if let Err(e) = res {
                        lane.error = e.message;
                    }
                    lane.done = true;
                }
                me.trace("compare", json!({"model": provider.model, "prompt": prompt, "ms": started.elapsed().as_millis() as u64}));
                (me.repaint)();
            });
        }
        (self.repaint)();
        Compare { run, stop }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_is_tokens_over_writing_time() {
        let l = Lane { tokens: 100, first_ms: 1000, total_ms: 3000, ..Default::default() };
        assert!((l.tokens_per_sec() - 50.0).abs() < 0.01);
        assert_eq!(Lane::default().tokens_per_sec(), 0.0);
    }
}
