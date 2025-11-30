# 🚀 Quick Start Guide - EnerSense Grid Frontend

## Installation & Run (3 Steps)

```bash
# 1. Install dependencies
npm install

# 2. Start development server
npm run dev

# 3. Open browser
# Navigate to http://localhost:5173
```

That's it! The app will be running locally.

## 📋 What You Get

✅ **4 Complete Pages:**
- Dashboard with real-time metrics and charts
- Transformer detail views with risk analysis
- Alerts management with filtering
- Analytics dashboard with multiple charts

✅ **Modern UI Components:**
- Responsive sidebar navigation
- Risk badges with color coding
- Interactive charts (Recharts)
- Glass morphism design
- Dark mode by default

✅ **Mock Data Included:**
- 6 transformers with realistic data
- Predictions and forecasts
- Alert history
- Analytics datasets

## 🎯 Key Features

### Dashboard
- Live transformer health summary
- Weather impact scoring
- Solar offset tracking
- Interactive risk map
- 6-hour load predictions
- System alerts preview

### Transformer Details
- Risk score visualization (0-100%)
- 6-hour load forecast with anomaly detection
- Cause breakdown analysis
- Recommended actions panel

### Alerts
- Filter by severity (High/Medium/Low)
- Real-time alert counts
- Direct links to transformer details

### Analytics
- 30-day risk trends
- Failure probability by transformer
- Solar vs Load comparisons
- Heat index influence graphs
- Overload prevention statistics

## 🛠️ Tech Stack

- **React 18** - UI Framework
- **Vite** - Build Tool (Fast!)
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Recharts** - Charts
- **Lucide Icons** - Icons

## 📁 File Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components
├── data/          # Mock JSON data
├── App.jsx        # Main app & routing
└── main.jsx       # Entry point
```

## 🔧 Customization

**Change colors?** Edit `tailwind.config.js`

**Add new page?** Create in `src/pages/` and add route in `App.jsx`

**Connect real API?** Replace JSON imports with fetch calls

## 📦 Build for Production

```bash
npm run build
# Output in dist/ folder
```

## 🐛 Troubleshooting

**Port already in use?**
```bash
npm run dev -- --port 3000
```

**Module errors?**
```bash
rm -rf node_modules package-lock.json
npm install
```

## ✨ Next Steps

1. **Connect Backend API** - Replace mock JSON with real API calls
2. **Add Authentication** - Implement login/user management
3. **Real-time Updates** - Add WebSocket connections
4. **Map Integration** - Replace placeholder with Leaflet/Mapbox
5. **Notifications** - Add browser notifications for alerts

---

**Ready to deploy?** Push to GitHub and connect to Vercel/Netlify for instant deployment!

