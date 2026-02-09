# Google Custom Search API Setup Guide

## Overview

This guide will help you set up Google Custom Search API for the Research Assistant's web search functionality.

## Prerequisites

- Google account
- Credit card (for Google Cloud Platform verification - **free tier available**)

## Free Tier Limits

- **100 searches per day** for free
- Beyond 100 searches: $5 per 1000 queries

## Step-by-Step Setup

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Create Project"** or select an existing project
3. Give your project a name (e.g., "Research Assistant")
4. Click **"Create"**

### 2. Enable Custom Search API

1. In the Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for **"Custom Search API"**
3. Click on **"Custom Search API"**
4. Click **"Enable"**

### 3. Create API Key

1. Go to **APIs & Services** → **Credentials**
2. Click **"Create Credentials"** → **"API Key"**
3. Copy the API key (you'll need this later)
4. (Optional) Click **"Restrict Key"** to limit it to Custom Search API only

### 4. Create Programmable Search Engine

1. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Click **"Add"** or **"Get Started"**
3. Configure your search engine:
   - **Sites to search**: Enter `www.google.com` (or leave blank to search entire web)
   - **Name**: Give it a name (e.g., "Research Assistant Search")
   - **Language**: Select your preferred language
4. Click **"Create"**
5. On the next page, click **"Customize"**
6. In the **"Basics"** section, find your **Search engine ID** (starts with a long alphanumeric string)
7. Copy the Search Engine ID

### 5. Configure Research Assistant

1. Open your `.env` file in the research-assistant directory
2. Replace the placeholder values:

```env
# Google Custom Search API
GOOGLE_SEARCH_API_KEY=AIzaSyC...your_actual_api_key_here
GOOGLE_SEARCH_ENGINE_ID=a1b2c3d4e5f6g7h8i...your_actual_engine_id_here
```

3. Save the file

### 6. Verify Setup

Run the verification script:

```bash
.venv\Scripts\activate
python -c "from src.tools.web_search import WebSearchTool; tool = WebSearchTool(); print(tool._run('test query'))"
```

If configured correctly, you should see search results (or a helpful error message if credentials are missing).

## Troubleshooting

### "API key not configured"
- Check that `GOOGLE_SEARCH_API_KEY` is set in your `.env` file
- Make sure there are no quotes around the key
- Restart your Python kernel/notebook if running

### "Search Engine ID not configured"
- Check that `GOOGLE_SEARCH_ENGINE_ID` is set in your `.env` file
- Make sure you copied the full ID from the Programmable Search Engine page

### "Invalid API key"
- Verify the API key is correct in Google Cloud Console
- Make sure Custom Search API is enabled
- Check if the API key has restrictions that might block requests

### "Quota exceeded"
- You've used your 100 free searches for the day
- Wait until tomorrow (quota resets at midnight Pacific Time)
- Or upgrade to a paid plan in Google Cloud Console

## Usage in Research Assistant

Once configured, the web search tool will automatically use Google Custom Search:

```python
from src.main import ResearchAssistant

assistant = ResearchAssistant()
assistant.load_documents(["your_document.pdf"])
assistant.setup_agent()

# Agent will use Google Search when needed
result = assistant.ask_agent("What are the latest AI developments in 2024?")
```

## Cost Management

To avoid unexpected charges:

1. **Set up billing alerts** in Google Cloud Console
2. **Monitor usage** in the Custom Search API dashboard
3. **Use quota limits** to cap daily searches
4. Remember: **100 searches/day are FREE**

## Additional Resources

- [Custom Search API Documentation](https://developers.google.com/custom-search/v1/overview)
- [Programmable Search Engine Help](https://support.google.com/programmable-search/)
- [Google Cloud Pricing](https://cloud.google.com/custom-search/pricing)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your credentials in Google Cloud Console
3. Check the error messages - they provide specific guidance
