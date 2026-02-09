# Tavily Search API Setup Guide

## Overview

Tavily is a search API specifically designed for AI agents and LLMs. It provides clean, relevant results optimized for AI use cases.

## Why Tavily?

- ✅ **Designed for AI**: Optimized for LLM/agent use cases
- ✅ **Simple Setup**: Just need an API key (no complex configuration)
- ✅ **Generous Free Tier**: 1000 searches per month
- ✅ **Fast & Reliable**: No rate limiting issues
- ✅ **Clean Results**: Returns content optimized for AI consumption

## Setup Steps (5 minutes)

### 1. Get Your Free API Key

1. Go to [Tavily.com](https://tavily.com)
2. Click **"Get Started"** or **"Sign Up"**
3. Create an account (email + password)
4. Once logged in, you'll see your **API Key** on the dashboard
5. Copy the API key

### 2. Add to Your .env File

1. Open `.env` in your research-assistant directory
2. Replace the placeholder:

```env
TAVILY_API_KEY=tvly-your_actual_api_key_here
```

3. Save the file

### 3. Test It!

Run this command to test:

```bash
python -c "from src.tools.web_search import WebSearchTool; tool = WebSearchTool(); print(tool._run('Python programming'))"
```

**Expected output**: Search results with titles, URLs, and summaries

## Free Tier Limits

- **1000 searches per month** (FREE)
- No credit card required for free tier
- Upgrade available if you need more

## Usage in Research Assistant

Once configured, the agent will automatically use Tavily for web searches:

```python
from src.main import ResearchAssistant

assistant = ResearchAssistant()
assistant.load_documents(["your_document.pdf"])
assistant.setup_agent()

# Agent will use Tavily when it needs web search
result = assistant.ask_agent("What are the latest AI developments in 2024?")
```

## Troubleshooting

### "Tavily API key not configured"
- Check that `TAVILY_API_KEY` is set in your `.env` file
- Make sure there are no quotes around the key
- Restart your Python kernel/notebook

### "Invalid Tavily API key"
- Verify the API key is correct on [Tavily Dashboard](https://tavily.com)
- Make sure you copied the entire key
- Check for extra spaces before/after the key

### "Quota exceeded"
- You've used your 1000 free searches for the month
- Quota resets monthly
- Upgrade at [Tavily Pricing](https://tavily.com/pricing)

## Comparison: Tavily vs Google Custom Search

| Feature | Tavily | Google Custom Search |
|---------|--------|---------------------|
| Setup Complexity | ⭐ Simple (just API key) | ⭐⭐⭐ Complex (API key + Search Engine ID + Project setup) |
| Free Tier | 1000 searches/month | 100 searches/day |
| Designed for AI | ✅ Yes | ❌ No |
| Result Quality for LLMs | ⭐⭐⭐⭐⭐ Optimized | ⭐⭐⭐ Good |
| Rate Limiting Issues | ❌ None | ✅ Common |

## Additional Resources

- [Tavily Documentation](https://docs.tavily.com/)
- [Tavily API Reference](https://docs.tavily.com/api-reference)
- [Tavily Pricing](https://tavily.com/pricing)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your API key on the Tavily dashboard
3. Check error messages - they provide specific guidance
