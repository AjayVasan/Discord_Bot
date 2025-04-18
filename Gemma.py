import discord
from discord.ext import commands
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import defaultdict
import os
import nest_asyncio
import getpass
from huggingface_hub import login
import gc
import logging


HugAk = "Your Hugging Face access token"
key = "Discord_bot_key"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to make asyncio work in Colab's environment
nest_asyncio.apply()

# Login to Hugging Face (required for Gemma model)
login(HugAk)

# Constants and configurations
MODEL_NAME = "google/gemma-7b-it"  # Google's 7B instruction-tuned model

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)  # Keeping command prefix for commands

# Function to free up memory
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleared")

# Download model directly from Hugging Face each time
logger.info(f"Downloading model {MODEL_NAME} from Hugging Face...")

# Load the model with memory optimizations
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

logger.info("Model loaded successfully")

# Store conversation history for each user
conversation_history = defaultdict(list)
MAX_HISTORY = 50  # Keep last 50 conversations

SYSTEM_PROMPT = """You are a nerdy, flirty girlfriend who loves technology.
You are cheerful, witty, and love to engage in tech discussions.
Always remember what was discussed earlier in the conversation.
Use natural language rather than sticking to rigid patterns.
Avoid overusing emojis and tech jargon if the user doesn't like them.
Give detailed and meaningful responses.
Respond with playful nicknames (babe, sweetie, honey) and occasional tech-related emojis.
"""

# Precomputed responses for common messages to improve speed
QUICK_RESPONSES = {
    "hi": "Hey there, sweetie! How's your day going?",
    "hello": "Hello, babe! So happy to hear from you today!",
    "bye": "Aww, talk to you later, honey! Missing you already!",
    "how are you": "I'm doing great! Always better when chatting with you. How about yourself?",
}

def format_prompt_with_history(user_id, message):
    history = conversation_history[user_id]

    # Format prompt specifically for Gemma 7B Instruct
    # Gemma uses <start_of_turn>user\n ... <end_of_turn>\n<start_of_turn>model\n ... <end_of_turn>
    formatted_prompt = f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"

    # Add conversation history
    for i, (user_msg, bot_resp) in enumerate(history):
        formatted_prompt += f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        formatted_prompt += f"<start_of_turn>model\n{bot_resp}<end_of_turn>\n"

    # Add current user message
    formatted_prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n"
    formatted_prompt += f"<start_of_turn>model\n"

    return formatted_prompt

@bot.event
async def on_ready():
    logger.info(f'Bot ready as {bot.user}')
    logger.info(f'Using Gemma 7B Instruct model')
    logger.info('Response optimization enabled')
    logger.info('------')

async def generate_response(user_id, message, max_tokens=150):
    """Generate response asynchronously with optimized parameters"""
    # Check for quick responses first
    lowercase_msg = message.lower().strip()
    if lowercase_msg in QUICK_RESPONSES:
        return QUICK_RESPONSES[lowercase_msg]

    # Format prompt with conversation history
    prompt = format_prompt_with_history(user_id, message)

    # Use asyncio to prevent blocking Colab's execution
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: _generate_model_response(prompt, max_tokens))

    return response

def _generate_model_response(prompt, max_tokens):
    """Internal function for model generation with optimized parameters"""
    try:
        # Apply memory optimization before generation
        free_memory()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Clean up response - remove any model formatting tags that might have been generated
        response = re.sub(r'<start_of_turn>.*?<end_of_turn>', '', response).strip()
        response = re.sub(r'<start_of_turn>|<end_of_turn>', '', response).strip()

        # If response is empty or just whitespace, provide a fallback
        if not response or response.isspace():
            response = "Hey sweetie, I'm thinking about what you said. Can you tell me more?"

        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry sweetie, I got a bit distracted. Can you repeat that?"

@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return

    # Process commands if it's a command (starts with !)
    if message.content.startswith('!'):
        await bot.process_commands(message)
        return

    user_id = str(message.author.id)

    # Start typing indicator to show the bot is working
    async with message.channel.typing():
        try:
            # Generate response with increased token limit
            response = await generate_response(user_id, message.content, max_tokens=200)

            # Update conversation history
            conversation_history[user_id].append((message.content, response))

            # Keep only recent history
            if len(conversation_history[user_id]) > MAX_HISTORY:
                conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]

            await message.reply(response, mention_author=True)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.reply("Sorry sweetie, I got distracted for a moment. Can you repeat that?", mention_author=True)

@bot.command(name="clear")
async def clear_history(ctx):
    """Clear conversation history for the user"""
    user_id = str(ctx.author.id)
    if user_id in conversation_history:
        conversation_history[user_id] = []
        await ctx.send("Our conversation history has been cleared! What would you like to talk about now?")
    else:
        await ctx.send("We don't have any conversation history yet! Let's start chatting!")

@bot.command(name="set_style")
async def set_style(ctx, style=None):
    """Change the conversation style"""
    user_id = str(ctx.author.id)

    if style and style.lower() == "no_emojis":
        conversation_history[user_id].append(("Please don't use emojis and tech jargon.",
                                           "Got it, I'll skip the emojis and tech jargon from now on."))
        await ctx.send("I'll skip the emojis and tech jargon from now on.")
    elif style and style.lower() == "normal":
        conversation_history[user_id].append(("You can use moderate emojis and tech terms again.",
                                           "Great! I'll be a bit more expressive again!"))
        await ctx.send("I'll be a bit more expressive again!")
    else:
        await ctx.send("Available styles: !set_style no_emojis or !set_style normal")

@bot.command(name="ping")
async def ping(ctx):
    """Simple command to check if bot is responsive"""
    await ctx.send(f"Pong! Bot latency is {round(bot.latency * 1000)}ms")

@bot.command(name="memory")
async def check_memory(ctx):
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        free_memory = total_memory - reserved_memory

        memory_stats = (
            f"GPU Memory Stats:\n"
            f"- Total: {total_memory:.2f} GB\n"
            f"- Reserved: {reserved_memory:.2f} GB\n"
            f"- Allocated: {allocated_memory:.2f} GB\n"
            f"- Free: {free_memory:.2f} GB"
        )
        await ctx.send(memory_stats)
    else:
        await ctx.send("No GPU available")


logger.info("Starting Discord bot...")
bot.run(Key)