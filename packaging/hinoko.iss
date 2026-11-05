; Inno Setup script -> Hinoko-Setup.exe (per-user install, no admin needed)
#define AppVersion GetEnv("HINOKO_VERSION")
#if AppVersion == ""
  #define AppVersion "0.15.1"
#endif

[Setup]
AppId={{6B1F2C0E-5A44-4C0B-9E27-0A1B2C3D4E5F}
AppName=Hinoko
AppVersion={#AppVersion}
AppPublisher=Hinoko
DefaultDirName={localappdata}\Programs\Hinoko
DefaultGroupName=Hinoko
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Hinoko-Setup
SetupIconFile=..\assets\hinoko.ico
UninstallDisplayIcon={app}\hinoko.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\target\release\hinoko.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\hinoko.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Hinoko"; Filename: "{app}\hinoko.exe"; IconFilename: "{app}\hinoko.ico"
Name: "{autodesktop}\Hinoko"; Filename: "{app}\hinoko.exe"; IconFilename: "{app}\hinoko.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\hinoko.exe"; Description: "Open Hinoko"; Flags: nowait postinstall skipifsilent
; an update installs silently: start Hinoko again afterwards
Filename: "{app}\hinoko.exe"; Flags: nowait skipifnotsilent
