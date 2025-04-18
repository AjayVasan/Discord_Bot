# Tech GF - Discord AI Companion Bot

![Bot Status](https://img.shields.io/badge/status-online-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Discord bot powered by Mistral-7B-Instruct-v0.2 that creates an engaging tech-savvy AI companion experience with a flirty personality, humor, and tech knowledge.

## 🌟 Features

- **Engaging Personality**: Tech-savvy girlfriend persona with humor, flirtation, and genuine conversations
- **Meme Integration**: Randomly includes relevant internet culture references and memes
- **Tech Knowledge**: Deep understanding of technology, gaming, and programming concepts
- **Conversation Memory**: Remembers previous conversations for cohesive interactions
- **Customizable Styles**: Change conversational style with simple commands
- **Optimized Performance**: Uses 4-bit quantization for efficient resource usage

## 🔗 Add to Your Server

Click the link below to add Tech GF to your Discord server:

[Add Tech GF to Discord](https://discord.com/oauth2/authorize?client_id=1362754862768459846&permissions=75776&integration_type=0&scope=bot)

## 💬 Commands

| Command | Description |
|---------|-------------|
| `!clear` | Clear your conversation history with the bot |
| `!set_style normal` | Enable regular conversation style with emojis and tech terms |
| `!set_style no_emojis` | Disable emojis and reduce tech jargon |
| `!set_style extra_flirty` | Enable a more flirtatious conversation style |
| `!set_style meme_queen` | Increase meme references and internet humor |
| `!ping` | Check if the bot is responsive and view latency |
| `!memory` | View the bot's current GPU memory usage |
| `!joke` | Get a random tech or programming joke |

## 🧠 Personality Traits

Tech GF has been designed with a rich personality including:

- Flirty and affectionate communication style
- Deep tech knowledge with clear explanations
- Playful humor with meme references
- Gaming enthusiasm and cyberpunk influences
- Emotional intelligence for genuine connections
- Competitive and challenging when appropriate

## 🛠️ Tech Stack

- **Language Model**: Mistral-7B-Instruct-v0.2
- **Framework**: discord.py
- **Optimization**: BitsAndBytes 4-bit quantization
- **Hosting**: GPU-accelerated cloud instance

## 🚀 Self-Hosting

1. Clone the repository:
```bash
git clone https://github.com/AjayVasan/Discord_Bot.git
cd Discord_Bot
```

2. Install dependencies:
```bash
pip install discord.py transformers torch huggingface_hub nest_asyncio accelerate bitsandbytes
```

3. Set up your environment:
   - Create a Discord bot and get your token from [Discord Developer Portal](https://discord.com/developers/applications)
   - Get a Hugging Face token from [Hugging Face](https://huggingface.co/settings/tokens)

4. Run the bot:
```bash
python bot.py
```

## 📝 Customization

You can customize the bot's personality by modifying:
- `SYSTEM_PROMPT` in the code to change the core personality
- `MEMES_AND_JOKES` list to add custom meme references
- `QUICK_RESPONSES` dictionary for custom replies to common messages

## 📊 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM

## ⚠️ Disclaimers

- This bot uses a large language model that may occasionally produce unexpected responses
- The flirty personality is designed to be playful and tasteful, not explicit
- Responses are AI-generated and should be treated as such

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue on GitHub.

## 👨‍💻 Author

- [Ajay Vasan](https://github.com/AjayVasan)

## 🙏 Acknowledgements

- Mistral AI for the base language model
- Hugging Face for model hosting
- The Discord.py team
