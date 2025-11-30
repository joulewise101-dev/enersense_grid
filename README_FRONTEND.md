# EnerSense Grid - Frontend

A modern, clean React application for AI-powered transformer load prediction and risk monitoring in Indian towns and villages.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open in browser:**
   Navigate to `http://localhost:5173` (Vite default port)

### Build for Production

```bash
npm run build
```

The production build will be in the `dist` folder.

## 📁 Project Structure

```
enersense-grid/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Navbar.jsx
│   │   ├── Sidebar.jsx
│   │   ├── RiskBadge.jsx
│   │   ├── TransformerCard.jsx
│   │   ├── AlertCard.jsx
│   │   ├── LineChart.jsx
│   │   ├── MapPlaceholder.jsx
│   │   └── RecommendationPanel.jsx
│   ├── pages/               # Page components
│   │   ├── Dashboard.jsx
│   │   ├── Transformers.jsx
│   │   ├── TransformerDetail.jsx
│   │   ├── Alerts.jsx
│   │   └── Analytics.jsx
│   ├── data/                # Mock JSON data
│   │   ├── transformers.json
│   │   ├── predictions.json
│   │   ├── alerts.json
│   │   └── analytics.json
│   ├── App.jsx              # Main app with routing
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── package.json
├── vite.config.js
├── tailwind.config.js
└── index.html
```

## 🎨 Features

### Pages

1. **Dashboard** (`/`)
   - Transformer health summary cards
   - Weather impact and solar offset metrics
   - Interactive risk map
   - 6-hour load prediction chart
   - System alerts preview
   - Grid of all transformers

2. **Transformers** (`/transformers`)
   - List view of all transformers
   - Search functionality
   - Click to view details

3. **Transformer Detail** (`/transformer/:id`)
   - Risk score visualization
   - 6-hour load prediction with anomaly markers
   - Cause breakdown (Heat Index, Festival Load, Solar Offset, Historical Load)
   - Recommended actions panel
   - Real-time statistics

4. **Alerts** (`/alerts`)
   - Filterable alert list (All, High, Medium, Low)
   - Alert summary statistics
   - Click alerts to view transformer details

5. **Analytics** (`/analytics`)
   - 30-day risk trend chart
   - Transformer failure probability graph
   - Solar vs Load 24-hour comparison
   - Heat index influence on load
   - Overloads prevented statistics

### Design System

**Colors:**
- Slate Gray (`#1E293B`) - Primary background
- Electric Blue (`#3B82F6`) - Primary accent
- Lime Green (`#22C55E`) - Success/Low risk
- Soft Yellow (`#FACC15`) - Warning/Medium risk
- Red (`#ef4444`) - High risk
- Light Gray (`#F1F5F9`) - Text/secondary

**Components:**
- Dark mode by default
- Glass morphism effects
- Smooth transitions and hover states
- Responsive grid layouts
- Interactive charts using Recharts

## 📊 Data Structure

All data is currently mock JSON files. To connect to a real API:

1. Create an API service in `src/services/api.js`
2. Replace JSON imports with API calls
3. Update components to handle loading states

Example:
```jsx
// src/services/api.js
export const fetchTransformers = async () => {
  const response = await fetch('/api/transformers')
  return response.json()
}
```

## 🔧 Customization

### Adding a New Page

1. Create component in `src/pages/YourPage.jsx`
2. Add route in `src/App.jsx`:
   ```jsx
   <Route path="/your-page" element={<YourPage />} />
   ```
3. Add navigation item in `src/components/Sidebar.jsx`

### Modifying Colors

Edit `tailwind.config.js` to change the color palette.

### Adding Charts

The project uses [Recharts](https://recharts.org/). See examples in:
- `src/components/LineChart.jsx`
- `src/pages/Analytics.jsx`

## 📦 Dependencies

- **React 18** - UI framework
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first CSS
- **Recharts** - Chart library
- **Lucide React** - Icon library
- **Vite** - Build tool

## 🐛 Troubleshooting

**Port already in use:**
```bash
# Change port in vite.config.js or use:
npm run dev -- --port 3000
```

**Module not found errors:**
```bash
# Clear node_modules and reinstall:
rm -rf node_modules package-lock.json
npm install
```

## 📝 Notes

- All data is currently mock/static JSON
- Charts use sample data arrays
- Map is a placeholder component (can be replaced with Leaflet/Mapbox)
- No authentication implemented (add if needed)

## 🚢 Deployment

### Vercel/Netlify
Simply connect your Git repository and deploy. Vite builds automatically.

### Manual Deployment
```bash
npm run build
# Upload dist/ folder to your hosting service
```

---

Built with ❤️ for EnerSense Grid

