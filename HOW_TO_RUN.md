# 🚀 How to Run EnerSense Grid Frontend

## Step-by-Step Instructions

### Step 1: Install Dependencies (First Time Only)

Open your terminal in the project folder and run:

```bash
npm install
```

This will install all required packages (React, Tailwind, Vite, etc.). Wait for it to complete.

### Step 2: Start the Development Server

Run this command:

```bash
npm run dev
```

You should see output like:
```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open in Browser

Click the link or manually navigate to:
```
http://localhost:5173
```

The app will automatically open in your browser! 🎉

---

## 📋 Quick Commands Reference

```bash
# Install dependencies (first time)
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## ⚠️ Troubleshooting

### "npm: command not found"
- Install Node.js from: https://nodejs.org/
- Restart your terminal after installation

### "Port 5173 already in use"
```bash
# Use a different port
npm run dev -- --port 3000
```

### "Module not found" errors
```bash
# Delete and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Dependencies installation stuck
- Try using: `npm install --legacy-peer-deps`
- Or use: `yarn install` (if you have yarn)

---

## ✅ What You Should See

Once running, you'll see:
- **Dark themed dashboard** with transformer cards
- **Sidebar navigation** (Dashboard, Transformers, Alerts, Analytics)
- **Interactive charts** and risk visualizations
- **Mock data** showing transformers, predictions, and alerts

---

## 🛑 To Stop the Server

Press `Ctrl + C` in the terminal where the server is running.

---

**That's it! Your EnerSense Grid frontend is now running!** ✨

