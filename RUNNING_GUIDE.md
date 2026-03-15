# Running Backend & Frontend - Complete Guide

This project has multiple ways to run it. Choose the setup that best fits your needs.

---

## Option 1: Tkinter Desktop App (Simplest - Recommended for Quick Testing)

**Best for**: Running locally with a desktop GUI, no API needed.

### PowerShell Steps:
```powershell
# Create Python virtual environment
py -3.11 -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt

# Run the app
python tkinter_app.py
```

**Access**: The Tkinter window opens directly on your machine
**Backend**: Not required
**Frontend**: Built-in Tkinter GUI

---

## Option 2: Backend API + React Web App

**Best for**: Full-stack development, web-based interface with map visualization.

### Prerequisites
- **Python 3.11+**
- **Node.js 16+** and npm

### Part A: Start the Backend API

In a terminal/PowerShell:

```powershell
# Navigate to project root
cd C:\Users\B\Desktop\ofek\classification-master\classification-master

# Create Python virtual environment
py -3.11 -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt

# Start the API server
python -m uvicorn app.main:app --reload --app-dir backend
```

**Backend will run on**: `http://127.0.0.1:8000`

**Useful endpoints:**
- `http://127.0.0.1:8000/docs` - Interactive API documentation (Swagger UI)
- `http://127.0.0.1:8000/redoc` - ReDoc API documentation

**Keep this terminal open** - the API needs to stay running.

---

### Part B: Start the React Web App

In a **new terminal/PowerShell** (keep Part A running):

```powershell
# Navigate to web app directory
cd web_app

# Install dependencies
npm install

# Start the dev server
npm run dev
```

**Web app will run on**: `http://localhost:5173` (or the URL shown in terminal)

**Features:**
- React 18 with Vite
- Leaflet map for geospatial visualization
- Connects to backend API for processing

**Keep this terminal open** - Vite dev server needs to stay running.

---

## Option 3: Electron Desktop App + Backend API

**Best for**: Standalone desktop app with web technology.

### Prerequisites
- **Python 3.11+**
- **Node.js 16+** and npm

### Part A: Start the Backend API

Follow "Option 2, Part A" above.

---

### Part B: Start Electron App

In a **new terminal/PowerShell** (keep backend running):

```powershell
# Navigate to electron app
cd app

# Install dependencies
npm install

# Start the dev server (runs both Vite + Electron)
npm run dev
```

**Features:**
- Electron desktop window
- Runs alongside Vite dev server
- Auto-reload on code changes

---

## Summary Table

| Option | Setup | Frontend | Backend | Best For |
|--------|-------|----------|---------|----------|
| **Tkinter** | Single command | Built-in | Not needed | Quick testing, offline |
| **Web App** | 2 terminals | React + Leaflet | FastAPI | Full web dev, maps |
| **Electron** | 2 terminals | Electron + React | FastAPI | Desktop app |

---

## Troubleshooting

### Backend won't start
```powershell
# Check Python version
py --version  # Should be 3.11+

# Reinstall dependencies
pip install -r backend/requirements.txt --upgrade

# Check if port 8000 is available
netstat -ano | findstr :8000
```

### Frontend won't connect to backend
- Ensure backend is running on `http://127.0.0.1:8000`
- Check CORS settings in `backend/app/main.py`
- Verify the proxy configuration in `vite.config.ts`

### npm install fails
```powershell
# Clear npm cache
npm cache clean --force

# Reinstall node modules
rm -r node_modules
npm install
```

### Venv activation fails
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Development Tips

- **Hot reload**: Both Vite and FastAPI `--reload` flag provide live reloading
- **API docs**: Visit `http://127.0.0.1:8000/docs` to test backend endpoints
- **Debug backend**: Check `.venv\Lib\site-packages` for installed packages
- **Debug frontend**: Open DevTools (F12) in browser for console errors

---

## Stopping Services

- **Backend**: Press `Ctrl+C` in the backend terminal
- **Frontend/Electron**: Press `Ctrl+C` in the frontend terminal
- **Tkinter app**: Close the window or press `Ctrl+C`

---

## Next Steps

1. Choose your preferred option above
2. Follow the steps in order
3. Check the troubleshooting section if you hit issues
4. Visit the API docs or web app URL to verify everything works
