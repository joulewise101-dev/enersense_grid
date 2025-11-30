# GitHub Pages Deployment Guide

Your EnerSense Grid app is now configured to work with GitHub Pages! Here's what was changed and how to deploy:

## Changes Made

1. **Switched to HashRouter**: Changed from `BrowserRouter` to `HashRouter` in `App.jsx` - this is required for GitHub Pages to handle routing correctly
2. **Updated Vite Config**: Added base path configuration for production builds
3. **Added Deployment Script**: Added `npm run deploy` command to package.json
4. **Created GitHub Actions Workflow**: Automatic deployment on push to main/master branch

## How to Deploy

### Option 1: Automatic Deployment (Recommended)

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Configure GitHub Pages"
   git push origin main
   ```

2. Enable GitHub Pages in your repository:
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under "Source", select **GitHub Actions**
   - Save the settings

3. The workflow will automatically deploy when you push to the main branch!

### Option 2: Manual Deployment

1. Install dependencies (if not already done):
   ```bash
   npm install
   ```

2. Build and deploy:
   ```bash
   npm run deploy
   ```

3. Enable GitHub Pages:
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under "Source", select the **gh-pages** branch
   - Save the settings

## Important Notes

- **URL Format**: Your site will be available at:
  - `https://YOUR_USERNAME.github.io/REPO_NAME/` (if repo is not your username.github.io)
  - `https://YOUR_USERNAME.github.io/` (if repo is your username.github.io)

- **Hash Routing**: URLs will now use hash routing (e.g., `#/transformers` instead of `/transformers`). This is necessary for GitHub Pages to work correctly.

- **First Deployment**: The first deployment may take a few minutes. After that, deployments are usually faster.

## Troubleshooting

If your site still doesn't work:

1. **Check GitHub Actions**: Go to the "Actions" tab in your repository to see if the deployment workflow ran successfully

2. **Verify Settings**: Make sure GitHub Pages is enabled and pointing to the correct source (GitHub Actions or gh-pages branch)

3. **Wait a few minutes**: Sometimes it takes 5-10 minutes for changes to propagate

4. **Clear browser cache**: Try opening the site in an incognito/private window

5. **Check the URL**: Make sure you're using the correct GitHub Pages URL format

## Testing Locally

You can test the production build locally before deploying:

```bash
npm run build
npm run preview
```

This will show you exactly how the site will look on GitHub Pages!

