# SmartAttend AI — Deploying to a permanent public URL

Your original notebook only ran through **ngrok**, which is why the link died
every time the Colab runtime stopped. This folder is a ready-to-deploy
version of the same app for **Streamlit Community Cloud** — it gives you a
free, permanent `https://your-app-name.streamlit.app` URL that stays up
without you keeping any notebook running.

## What's in here
- `app.py` — the Streamlit app (extracted from your notebook, unchanged logic)
- `requirements.txt` — Python packages (uses `dlib-bin`, a precompiled dlib
  wheel, so the cloud build doesn't have to compile dlib from source — that's
  the #1 reason face-recognition apps fail to deploy)
- `packages.txt` — system libraries Streamlit Cloud needs to install via apt
- `.streamlit/config.toml` — dark theme + server settings
- `.gitignore` — keeps the SQLite DB / uploaded photos out of git

## Step 1 — Push this folder to GitHub
1. Create a **new GitHub repository** (public or private both work), e.g. `smart-attendance-system`.
2. Upload every file in this folder to the repo, keeping the folder structure
   (the `.streamlit/config.toml` path matters — don't flatten it).
   - Easiest way: on github.com, click **Add file → Upload files**, drag in
     `app.py`, `requirements.txt`, `packages.txt`, `.gitignore`, and the
     `.streamlit` folder (or use `git` locally — see below).

   ```bash
   git init
   git add .
   git commit -m "Smart Attendance System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/smart-attendance-system.git
   git push -u origin main
   ```

## Step 2 — Deploy on Streamlit Community Cloud
1. Go to **https://share.streamlit.io** and sign in with GitHub (free).
2. Click **Create app → Deploy a public app from GitHub**.
3. Pick your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**.
5. First build takes ~5–10 minutes (it's compiling/installing the
   face-recognition stack). After that you get a permanent URL like:
   `https://smart-attendance-system.streamlit.app`

That URL works forever — no ngrok token, no expiring tunnel, no Colab
session needed. You can share it with anyone.

## Step 3 — First login
The app creates its own SQLite database on first run but ships with no
admin account. Open the app, go to the **Register/Sign up** tab on the auth
screen, and create your first admin account there.

## Important limitation: storage is not permanent
Streamlit Community Cloud's filesystem is **ephemeral** — it resets whenever
the app restarts (goes to sleep after inactivity, or you push a new commit).
That means:
- The SQLite database (`attendance.db`), uploaded student photos, and face
  encodings will be **wiped on restart**.
- This is fine for a demo/FYP presentation, but not for real long-term data.

If you need data to persist between restarts, the fix is to point the app at
an external database instead of local SQLite — options, roughly in order of
effort:
1. **Turso / Supabase / Neon** (free-tier hosted Postgres or SQLite-compatible DB)
2. **Streamlit's built-in `st.connection`** to a cloud DB
3. Store uploaded photos in an S3-compatible bucket (e.g. Cloudflare R2 free tier)

Say the word and I can wire the app up to one of these so attendance data
survives restarts — just tell me which you'd prefer (Postgres is the most
common choice).

## Alternative hosts (if you'd rather not use Streamlit Cloud)
- **Hugging Face Spaces** (free, supports Streamlit SDK directly, similarly
  ephemeral storage but slightly more generous compute)
- **Render.com** (free/paid web service, persistent disk available on paid
  tier, good if you need the DB to actually persist)

I can prepare the equivalent config for either of those if you tell me which
one you want.
