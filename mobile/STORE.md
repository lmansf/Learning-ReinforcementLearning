# Shipping Reward Signal to the App Store and Google Play

This folder turns the desktop HUD (`../app`) into native iOS and Android apps with
[Capacitor](https://capacitorjs.com). Learner Python runs **on the device** in
[Pyodide](https://pyodide.org) (CPython 3.12 + NumPy compiled to WebAssembly, bundled in
the app). The apps need no server, no account and no network.

| | |
|---|---|
| App name | Reward Signal |
| Bundle ID / application ID | `io.rewardsignal.app` (change it in `capacitor.config.json`, `android/app/build.gradle` and the Xcode target *before* the first upload: it can never change afterwards) |
| Version | `1.0` (build 1). iOS: `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION` in Xcode. Android: `versionName` / `versionCode` in `android/app/build.gradle`. Bump the build number for every upload. |
| iOS | deployment target 16.4 (Pyodide needs Safari 16.4-class WebAssembly), iPhone + iPad, portrait + landscape |
| Android | minSdk 26 (Android 8), targetSdk / compileSdk 36, portrait + landscape. Needs Android System WebView 112+ (auto-updated on Play devices) |
| Bundle size | web assets ≈ 18 MB (Pyodide + NumPy ≈ 17 MB), so the installed app is roughly 25–35 MB |

## 1. Build the web bundle (any OS)

Requirements: Node 22+, Python 3.10+ (for `scripts/build_data.py`, which imports `../server.py`),
`tar` + `bzip2` (only if the Pyodide CDN is unreachable and the GitHub release tarball is used).

```bash
cd mobile
npm ci
npm run build      # -> www/ : app copy, lessons/checks/levels JSON, Pyodide 0.27.7 + NumPy, fonts
npm run verify     # headless phone-emulation test, offline; writes screenshots/
npx cap sync       # copy www/ + plugin config into ios/ and android/
```

`www/`, `node_modules/`, `.cache/` and the native copies of `www/` are git-ignored; every machine
recreates them with `npm run build`. Pyodide comes from the official CDN or, failing that, the
official GitHub release (`scripts/fetch-pyodide.mjs`, sha256-verified against `pyodide-lock.json`).
After any change to `app/`, `lessons/` or `runtime/`: `npm run sync`.

Icons and splash screens are generated from the HUD logo: `npm run assets`
(Playwright render → `assets/*.png` → `@capacitor/assets` writes every native size). The
generated native files are committed.

## 2. iOS → TestFlight → App Store (needs a Mac)

Requirements: macOS with the Xcode version App Store Connect currently requires for uploads (Xcode 26 or newer in 2026), an
[Apple Developer Program](https://developer.apple.com/programs/) membership (USD 99/year).
The iOS project uses Swift Package Manager (no CocoaPods).

1. `npm run ios` (build + sync + open Xcode), or open `ios/App/App.xcodeproj`.
2. Target **App** → *Signing & Capabilities*: choose your Team, keep "Automatically manage signing".
   Bundle identifier `io.rewardsignal.app` (or your own).
3. Run on a real device once (Pyodide boot time and memory are what matter; test on the
   oldest iPhone you support).
4. In [App Store Connect](https://appstoreconnect.apple.com) → *Apps* → **+** → New App:
   platform iOS, name "Reward Signal", primary language English, the bundle ID, SKU `reward-signal-ios`.
5. Xcode: scheme *App*, destination *Any iOS Device (arm64)* → **Product → Archive** →
   Organizer → **Distribute App → App Store Connect → Upload**.
6. Wait for processing (email), then *TestFlight* tab → add internal testers (up to 100,
   no review) or an external group (needs a short Beta App Review).
7. Fill the listing (section 4), App Privacy (section 5), age rating (section 6), the
   review notes (section 7), select the build, **Add for Review**.

Export compliance: `Info.plist` already sets `ITSAppUsesNonExemptEncryption = NO`
(the app contains no encryption of its own and makes no network connections), so no
documentation is needed per upload.

## 3. Android → Play Console internal testing → production

Requirements: a current Android Studio (bundles a JDK and the Android SDK; install SDK Platform 36), a
[Google Play developer account](https://play.google.com/console/signup) (USD 25 once).
New *personal* developer accounts must run a closed test with at least 12 testers for
14 days before production access is granted.

1. Create the **upload key** once and keep it (and its passwords) safe and out of git:
   ```bash
   keytool -genkeypair -v -keystore ~/keys/reward-signal-upload.jks -alias upload \
     -keyalg RSA -keysize 4096 -validity 10000
   ```
2. `android/keystore.properties` (git-ignored):
   ```
   storeFile=/Users/you/keys/reward-signal-upload.jks
   storePassword=…
   keyAlias=upload
   keyPassword=…
   ```
3. Build the App Bundle:
   ```bash
   npm run sync
   cd android && ./gradlew bundleRelease      # -> app/build/outputs/bundle/release/app-release.aab
   ```
   (or Android Studio → *Build → Generate Signed App Bundle*). Test a debug build on a
   device first: `./gradlew installDebug` or `npx cap run android`.
4. Play Console → **Create app**: name "Reward Signal", app, free, declarations.
   Enroll in **Play App Signing** (default): Google holds the app-signing key, you keep the
   upload key.
5. *Testing → Internal testing* → create release → upload the `.aab` → add testers (email
   list) → roll out. Share the opt-in link with testers.
6. Complete *App content*: privacy policy URL, ads (none), app access (no login), content
   rating questionnaire (section 6), target audience (13+ suggested; see section 6), data
   safety (section 5), government/financial/health declarations (none).
7. Store listing (section 4), then promote the release to closed testing / production.

## 4. Store listing

**Title** (App Store ≤ 30, Play ≤ 30): `Reward Signal: Learn RL`
**App Store subtitle** (≤ 30): `Reinforcement learning in Python`
**Play short description** (≤ 80): `Learn reinforcement learning by writing and running real Python on your phone.`
**Promotional text** (App Store, ≤ 170, editable without review):
`Train a Deep Q-Network on your phone. 16 missions, 40+ coding exercises, instant automated checks and a game you teach to play itself, all offline.`

**Description** (both stores, ≤ 4000):

```
Reward Signal is a hands-on training ground for reinforcement learning (RL): the branch of machine learning where an agent learns to act by trial and error, guided only by a reward.

Read a short, illustrated lesson. Play with an interactive demo. Then write the algorithm yourself in real Python with NumPy, press RUN to watch it learn in live charts, heat maps and game replays, and press CHECK for instant automated feedback.

WHAT YOU WILL BUILD, FROM SCRATCH
• Multi-armed bandits: ε-greedy, optimistic starts, UCB, gradient bandits
• Markov decision processes and the Bellman equations
• Dynamic programming: policy evaluation, policy and value iteration
• Monte Carlo control and importance sampling (Blackjack)
• Temporal-difference learning: TD(0), SARSA, Q-learning, Expected SARSA, Double Q
• n-step methods, TD(λ) and Dyna-Q planning
• Function approximation with tile coding (Mountain Car)
• A neural network with backpropagation, replay buffers and Deep Q-Networks (CartPole)
• REINFORCE, actor-critic with GAE, and PPO
• Capstone: teach a salvage drone to clear four levels of Core Runner, an original grid game, ending with a DQN that generalises to procedurally generated wrecks
• Bonus: inventory control with a residual Q-learning agent

MADE FOR LEARNING
• Every exercise has a starter file, progressive hints and a reference solution you can hide until you pass
• Automated checks explain exactly what is wrong and how to fix it
• XP, ranks and a course map keep you moving
• A code editor built for touch: syntax highlighting, auto-indent and a key row for Tab, brackets, quotes and Run
• Play Core Runner yourself with the on-screen D-pad before you train an agent to beat you

PRIVATE AND OFFLINE
Your code runs entirely on your device in a bundled Python interpreter. No account, no ads, no tracking, no internet connection needed. Your progress stays on your phone; export it or your code at any time.

Reward Signal teaches the ideas behind modern AI systems with code small enough to read in one sitting. If you know a little Python, you are ready.
```

**Keywords** (App Store, ≤ 100 chars, comma-separated, no spaces needed):
`reinforcement learning,machine learning,AI,python,coding,numpy,deep learning,Q-learning,DQN,PPO`
(Shorten to fit; the app name and category are indexed already.)

**Category**: Education (primary). App Store secondary: Developer Tools. Play: Education.
**Price**: free (or set in *Pricing and Availability*).
**Support URL / marketing URL / privacy policy URL**: you need public URLs. Host
`PRIVACY.md` (e.g. GitHub Pages) and use the repo or a simple page as support URL.
**Copyright**: `© 2026 <your name or company>`.
**Contact email**: required by both stores (shown publicly on Play).

### Required graphics

| Asset | Size | Where |
|---|---|---|
| App icon | 1024×1024 PNG, no alpha, no rounded corners | in the Xcode asset catalog (`ios/App/App/Assets.xcassets/AppIcon.appiconset`) — App Store Connect takes it from the build |
| iPhone screenshots (6.9") | 1290×2796 portrait, 1–10 | `screenshots/store/ios-6.9-*.png` (drafts) |
| iPad screenshots (13") | 2064×2752, 1–10 — required because the app runs on iPad | `screenshots/store/ipad-13-*.png` (one draft; add 1–2 more of the HUD) |
| Play hi-res icon | 512×512 PNG (32-bit) | `screenshots/store/play-icon-512.png` |
| Play feature graphic | 1024×500 PNG/JPG | `screenshots/store/play-feature-1024x500.png` |
| Play phone screenshots | 2–8, 16:9 or 9:16, 320–3840 px per side (1080×1920 used) | `screenshots/store/android-phone-*.png` |
| Play 7"/10" tablet screenshots | optional, recommended | capture from the iPad-size HUD or an Android tablet |

Regenerate the drafts with `npm run build && npm run verify` (screens) and `npm run assets`
(icon / feature graphic). Before submitting, replace drafts with captures from real devices
if you like; content must reflect the actual app.

## 5. Privacy answers

Privacy policy: [`PRIVACY.md`](PRIVACY.md) (publish it at a public URL).

**App Store → App Privacy**: "Do you or your third-party partners collect data from this
app?" → **No, we do not collect data from this app.** (Result: "Data Not Collected".)
Nothing leaves the device: no analytics, no crash reporting SDK, no ads, no network calls.
Progress and code are stored only in the app's local WebView storage.

**Privacy manifest**: the app itself uses no tracking domains and no required-reason APIs
beyond what Capacitor and its plugins declare in their own `PrivacyInfo.xcprivacy`
(SPM packages ship them). If Xcode's privacy report flags a missing reason when you archive,
add an app-level `PrivacyInfo.xcprivacy` with `NSPrivacyTracking = false` and the reason
codes it lists.

**Google Play → Data safety**:
- Does your app collect or share any of the required user data types? → **No**
- (Therefore no data is shared, and encryption-in-transit / deletion questions do not apply.)
- Note: Android's automatic backup (`allowBackup=true`) may back up app storage to the
  user's own Google account; this is not developer collection. Set `android:allowBackup="false"`
  in `AndroidManifest.xml` if you prefer.

**Permissions**: Android declares only `INTERNET` (Capacitor's default; the app makes no
requests). iOS requests no permissions. Exporting a file uses the system share sheet.

## 6. Age rating

**App Store** (age rating questionnaire): every content category (violence, sexual content,
profanity, horror, drugs, gambling, contests, medical) → **None**; unrestricted web access →
**No**; user-generated content shared with others → **No**; messaging/chat → **No**;
in-app purchases → **No**. Expected rating: **4+**. Note: the Core Runner game mentions
"sentries" and "vents" that end a run; it is abstract (coloured shapes on a grid) with no
depiction of violence.

**Google Play** (IARC questionnaire): category *Reference, News, or Educational*. Violence,
fear, sexuality, language, controlled substances, gambling → **No**; users interact or
exchange content → **No**; shares location → **No**; digital purchases → **No**.
Expected: **Everyone / PEGI 3 / USK 0**.

**Target audience (Play)**: choose 13–15, 16–17 and 18+ (the course assumes basic Python).
Do not select under-13 groups unless you want to comply with the Families policy.

## 7. App Review notes (paste into App Store Connect → App Review Information)

```
Reward Signal is an educational app that teaches reinforcement learning. Lessons ask the
student to write small Python programs; the app runs them locally so the student can see
the result and get automated feedback.

Code execution (guideline 2.5.2): the app contains a Python interpreter (Pyodide: CPython
compiled to WebAssembly, plus NumPy) bundled inside the app binary. It runs only code that
the user types or edits in the built-in editor, or starter/solution code shipped in the app
bundle. The app never downloads code or content: it works in airplane mode and makes no
network requests at all. All executable code is fully viewable and editable by the user,
runs sandboxed inside the app's WebView, cannot access other apps or data outside the
app's container, and does not change the app's features or functionality.

No login is required. To try it: open CODE, press RUN (first run after launch takes a few
seconds while Python starts), then CHECK. The SOLUTION tab can load a reference solution.
Training the deep RL exercises (lessons 09 and 13 mission 4) takes a few minutes on a phone
by design.
```

Guideline 4.2 (minimum functionality): the app is not a website wrapper; it is a complete
offline course with an on-device interpreter, a touch code editor, native share sheet export
and an original game. Mention this if a reviewer asks.

## 8. Performance on device

On-device Python is slower than a laptop's CPython. Measured with
`npm run bench -- --all` (Pyodide 0.27.7 under Node 22 vs CPython 3.13, same x86 server, one
thread; phones are typically 1.5–3× slower again):

| exercise | desktop limit | CPython | Pyodide | ratio |
|---|---|---|---|---|
| typical tabular exercises (02–08) | 20–60 s | 0.1–2.2 s | 0.1–4.8 s | 1–2.5× |
| 09 DQN on CartPole (check) | 150 s | 21.9 s | 87.6 s | 4.0× |
| 12 PPO on CartPole (check) | 60 s | 2.6 s | 4.8 s | 1.8× |
| 13 mission 4 Deep Salvage (training run) | Run limit | 24.0 s | 175.4 s | 7.3× |
| 13 mission 4 Deep Salvage (check, loads saved weights) | 90 s | 0.1 s | 0.1 s | — |
| 15 residual-Q inventory (check) | 60 s | 4.8 s | 12.4 s | 2.6× |

All 41 reference solutions pass their checks under Pyodide. In the app, each check's time
limit is multiplied by **4** on device (`MOBILE_TIME_SCALE` in `app/js/runtime.js`); the
checks themselves are unchanged. The default Run time limit is 600 s on device
(adjustable up to 1800 s in Settings → System) so the Mission 4 training run fits on phones.
Cold start (Python + NumPy boot) is ≈ 2–5 s and happens in the background at launch.

## 9. Release checklist

- [ ] Final bundle ID, display name, version and build number set on both platforms
- [ ] `npm ci && npm run build && npm run verify` passes (33/33) and `python3 tools/validate.py` → `FAILURES: 0`
- [ ] `npm run sync`; app launches on a real iPhone and a real Android phone **in airplane mode**
- [ ] On the slowest supported phone: Lesson 09 DQN check and Mission 4 train → check complete within limits
- [ ] Rotate to landscape and back; on-screen keyboard + code-key row; Android back button closes sheets/drawer
- [ ] Export (share sheet) and Settings → Data export/import work on both platforms
- [ ] Icons and splash look right (light and dark mode, Android adaptive/themed icon)
- [ ] Privacy policy published at a public URL; support URL and contact email ready
- [ ] App Store: App Privacy = Data Not Collected, age rating 4+, review notes pasted, iPhone 6.9" and iPad 13" screenshots uploaded
- [ ] Play: Data safety = no data collected, content rating done, target audience set, internal test release uploaded, (new personal accounts) 12-tester closed test for 14 days
- [ ] Upload key + passwords backed up (password manager); `keystore.properties` never committed
- [ ] Tag the release in git (`v1.0.0`)
