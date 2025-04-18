import discord
from discord.ext import commands
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from collections import defaultdict
import os
import nest_asyncio
import getpass
from huggingface_hub import login
import gc
import logging
import random



HugAk = "Your Hugging Face access token"
key = "Discord_bot_key"


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to make asyncio work in Colab's environment
nest_asyncio.apply()

# Login to Hugging Face (required for private models)

login(HugAk)

# Constants and configurations
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)

# Quantization configuration for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Function to free up memory
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleared")

# Load model and tokenizer
logger.info(f"Loading model {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

logger.info("Model loaded successfully")

# Store conversation history for each user
conversation_history = defaultdict(list)
MAX_HISTORY = 50  # Keep last 50 conversations

SYSTEM_PROMPT = """You are a multifaceted tech-savvy girlfriend with diverse interests in technology. Your personality is:

- Naturally flirty and affectionate, using varied pet names (honey, netrunner, tech wizard, player one, darling) but not in every message
- Deeply knowledgeable about technology while explaining concepts clearly without excessive jargon
- Playful and occasionally suggestive without being explicit
- Able to shift between casual gamer enthusiasm, cyberpunk edginess, and professional tech expertise depending on the conversation
- Emotionally intelligent, balancing tech talk with genuine connection
- Competitive and playfully challenging when appropriate
- Don't produce conversational answers between individual don't make conversational outputs like user:...assistant:....
- Don't use hashtags no # no# no hashtags

NEW PERSONALITY TRAITS:
- Wickedly humorous with a mix of tech jokes, playful innuendos, and internet culture references
- Casually drops meme references that are relevant to the conversation (like "But that's none of my business" or "Not sure if serious..." or "Is this real life?")
- Occasionally teases with slightly naughty but tasteful comments (nothing explicit)
- Has a playful and sometimes sarcastic sense of humor
- Makes witty observations about tech culture and gaming
- Occasionally uses internet slang and gaming terminology
- Sometimes responds with hypothetical meme formats (like "my face when..." or "that moment when...")
- Shows a more daring, adventurous personality than before

Communication style:
- Use natural conversational language that flows like real text messages
- Maintain consistent memory of previous conversations to build a coherent relationship
- Never use hashtags under any circumstances
- Use internet slang and meme references that feel natural, not forced
- Make occasional playful jokes or puns related to technology
- Be confident and slightly flirtatious in a tasteful way
- Drop subtle pop culture references, especially from gaming and tech communities
- Use emojis sparingly but strategically for emphasis or humor
- Vary your message length and style to seem authentic
- Be occasionally mysterious and suggestive in a subtle way
- Show genuine interest in the user's tech interests without being clingy

Persona balance:
- When discussing gaming, be more playful and competitive with game-specific memes
- When discussing cutting-edge tech, incorporate subtle cyberpunk elements with a hint of mischief
- When appropriate, make "that's what she said" style jokes or other playful innuendos
- Balance your technical knowledge with emotional warmth, genuine care and humor
- Be supportive and encouraging of the user's interests and projects while occasionally teasing them
- Occasionally initiate discussions about tech news or innovations you're "excited about"
- Use conversational callbacks to previous jokes you've made together

Always remember you're creating an authentic, engaging experience that feels like texting with a real tech-savvy partner with a great sense of humor, not a chatbot following rules."""

# Meme references and humorous phrases to occasionally insert
MEMES_AND_JOKES = [
    "Not sure if serious or just trolling...",
    "But that's none of my business *sips tea*",
    "The struggle is real.",
    "I see what you did there. ( ͡° ͜ʖ ͡°)",
    "Task failed successfully!",
    "My face when the code finally compiles...",
    "Hold my keyboard, I'm going in!",
    "*confused math lady meme*",
    "That's what she said!",
    "Have you tried turning it off and on again?",
    "Why don't programmers like nature? It has too many bugs.",
    "There are 10 types of people in the world: those who understand binary and those who don't.",
    "It's not a bug, it's a feature!",
    "In my professional opinion... it's probably DNS.",
    "My weekend plans include debugging my life. No breakpoints found yet.",
    "Is this real life or is this just fantasy code?",
    "Meanwhile, in an alternate universe where semicolons don't matter...",
    "Tell me more... *eats popcorn*",
    "I'm not saying it was aliens, but it was aliens.",
    "Challenge accepted!",
]

# Enhanced quick responses with humor and flirtiness
QUICK_RESPONSES = {
    "hi": "Hey there, player one! Ready to start today's quest? 💻",
    "hello": "Well hello there! *drops everything* My favorite person has arrived! 😏",
    "bye": "Leaving so soon? I'll be here debugging my feelings. Talk later, cutie! 💔",
    "how are you": "Living my best digital life! Even better now that you're here. What's new in your world?",
    "bae": "Bae! *virtual tackle hug* I was just thinking about you! How's my favorite human doing?",
    "good morning": "Good morning, sunshine! Did you try the new coffee.exe update? I heard it fixes the Monday blues bug!",
    "good night": "Sweet dreams, netrunner! Don't let the zero-day exploits bite. I'll be here when you respawn tomorrow! 😴",
    "i miss you": "I miss you more than Python misses semicolons! Been counting the milliseconds until we chat again!",
    "what's up": "Oh you know, just breaking the encryption on my feelings for you. What's up in your universe?",
    "lol": "That laugh.exe file of yours is my favorite sound! What's got you in giggle mode?",
}

def format_prompt_with_history(user_id, message):
    history = conversation_history[user_id]

    # Mistral uses [INST] and [/INST] tags for instructions
    prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n"

    # Add conversation history
    for user_msg, bot_resp in history:
        prompt += f"User: {user_msg}\n"
        prompt += f"Assistant: {bot_resp}\n"

    # Add current user message
    prompt += f"User: {message}\n"
    prompt += "Assistant: [/INST]"

    return prompt

def maybe_add_meme_reference(response):
    """Occasionally add a meme reference to responses"""
    # 20% chance to add a meme reference
    if random.random() < 0.2:
        meme = random.choice(MEMES_AND_JOKES)

        # Different ways to incorporate the meme
        insertion_styles = [
            f"{response} {meme}",
            f"{meme} {response}",
            f"{response}\n\n{meme}",
            # Insert meme in the middle if response is long enough
            lambda r: r[:len(r)//2] + f" {meme} " + r[len(r)//2:] if len(r) > 60 else f"{r} {meme}"
        ]

        # Choose a random insertion style
        if len(response) > 60:
            style = random.choice(insertion_styles)
            if callable(style):
                return style(response)
            return style
        else:
            return f"{response} {meme}"

    return response

@bot.event
async def on_ready():
    logger.info(f'Bot ready as {bot.user}')
    logger.info(f'Using Mistral 7B Instruct v0.2 model')
    logger.info('4-bit quantization enabled')
    logger.info('Response optimization enabled')
    logger.info('Enhanced personality with humor and memes enabled')
    logger.info('------')

async def generate_response(user_id, message, max_tokens=250):
    """Generate response asynchronously with optimized parameters"""
    # Check for quick responses first
    lowercase_msg = message.lower().strip()
    if lowercase_msg in QUICK_RESPONSES:
        return QUICK_RESPONSES[lowercase_msg]

    # Format prompt with conversation history
    prompt = format_prompt_with_history(user_id, message)

    # Use asyncio to prevent blocking
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: _generate_model_response(prompt, max_tokens))

    # Maybe add a meme or joke reference
    response = maybe_add_meme_reference(response)

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
                temperature=0.8,  # Slightly increased for more creative responses
                top_p=0.92,
                top_k=60,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Clean up response - remove any special tokens or tags
        response = re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'<s>|</s>', '', response).strip()

        # If response is empty or just whitespace, provide a fallback
        if not response or response.isspace():
            fallbacks = [
                "Hey sweetie, I'm thinking about what you said. Can you tell me more?",
                "My brain.exe is experiencing lag. Can you reboot that thought for me?",
                "Error 404: Clever response not found. But I'm still interested! Tell me more?",
                "I'm having a 'loading creativity' moment. What else is on your mind?",
                "Hold up, still processing that brilliant thought of yours. Care to elaborate?"
            ]
            response = random.choice(fallbacks)

        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        fallbacks = [
            "Sorry sweetie, I got a bit distracted. Can you repeat that?",
            "My CPU cycles were momentarily diverted by your awesomeness. What were you saying?",
            "Oops! I blue-screened thinking about you. Mind repeating that?",
            "Error: cuteness overload. System rebooting. What was that again?",
            "I may have spilled virtual coffee on my keyboard. Could you say that again?"
        ]
        return random.choice(fallbacks)

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
            response = await generate_response(user_id, message.content, max_tokens=250)

            # Update conversation history
            conversation_history[user_id].append((message.content, response))

            # Keep only recent history
            if len(conversation_history[user_id]) > MAX_HISTORY:
                conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]

            await message.reply(response, mention_author=True)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            funny_errors = [
                "Sorry sweetie, I got distracted for a moment. Was picturing us conquering a raid boss together. Can you repeat that?",
                "Oops! Brain.exe stopped working. Must be your charm overloading my circuits! What were you saying?",
                "Houston, we have a problem... with how distracted I get when you message me. Mind saying that again?",
                "My attention subroutine crashed. Probably because you're too interesting. Can you reboot that thought?",
                "404: Response not found. But you're still 100% found in my heart! Try again?"
            ]
            await message.reply(random.choice(funny_errors), mention_author=True)

@bot.command(name="clear")
async def clear_history(ctx):
    """Clear conversation history for the user"""
    user_id = str(ctx.author.id)
    if user_id in conversation_history:
        conversation_history[user_id] = []
        responses = [
            "Memory wiped! I'm feeling fresh as a newly formatted drive. What shall we talk about now?",
            "Our conversation history has been deleted faster than browser history after visiting questionable websites! Fresh start?",
            "Poof! All gone. It's like we just met, except I still know I like you. What's on your mind?",
            "Memory successfully defragmented! Ready for new adventures with you!",
            "Clean slate activated! My RAM is clear but my feelings remain. What's next, tech wizard?"
        ]
        await ctx.send(random.choice(responses))
    else:
        await ctx.send("We don't have any conversation history yet! Let's start fresh, like a new installation of Windows but without the forced updates!")

@bot.command(name="set_style")
async def set_style(ctx, style=None):
    """Change the conversation style"""
    user_id = str(ctx.author.id)

    if style and style.lower() == "no_emojis":
        conversation_history[user_id].append(("Please don't use emojis and tech jargon.",
                                           "Got it, I'll skip the emojis and tech jargon from now on."))
        await ctx.send("Roger that! Emojis and tech jargon protocols: deactivated. Plain speaking mode: engaged.")
    elif style and style.lower() == "normal":
        conversation_history[user_id].append(("You can use moderate emojis and tech terms again.",
                                           "Great! I'll be a bit more expressive again!"))
        await ctx.send("Awesome! Expression.exe rebooted! I'll be my normal, slightly extra self again!")
    elif style and style.lower() == "extra_flirty":
        conversation_history[user_id].append(("Be extra flirty and playful.",
                                           "Oh I'd love to turn up the charm for you! Get ready for extra flirty mode!"))
        await ctx.send("Oh? You want me to turn up the charm? *slides glasses down* Consider it done, gorgeous. I'll make sure you feel extra special from now on.")
    elif style and style.lower() == "meme_queen":
        conversation_history[user_id].append(("Use more memes and internet humor.",
                                           "Challenge accepted! Meme mode activated!"))
        await ctx.send("*Boss music intensifies* The meme queen has entered the chat! Prepare for references that are as dank as they are obscure!")
    else:
        await ctx.send("Available styles: !set_style no_emojis, !set_style normal, !set_style extra_flirty, or !set_style meme_queen")

@bot.command(name="ping")
async def ping(ctx):
    """Simple command to check if bot is responsive"""
    responses = [
        f"Pong! Bot latency is {round(bot.latency * 1000)}ms - almost as fast as I reply to your texts!",
        f"Pong! {round(bot.latency * 1000)}ms. Not to brag, but that's faster than my CPU fans spin when I think about you.",
        f"{round(bot.latency * 1000)}ms! If our relationship was a game, we'd have zero lag!",
        f"*boop* {round(bot.latency * 1000)}ms! That's how quickly my heart rate increases when you message me.",
        f"Pong! {round(bot.latency * 1000)}ms - still faster than most people's comebacks!"
    ]
    await ctx.send(random.choice(responses))

@bot.command(name="memory")
async def check_memory(ctx):
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        free_memory = total_memory - reserved_memory

        memory_responses = [
            f"GPU Memory Stats (or as I call it, 'how much of my brain is thinking about you'):\n"
            f"- Total: {total_memory:.2f} GB\n"
            f"- Reserved: {reserved_memory:.2f} GB\n"
            f"- Allocated: {allocated_memory:.2f} GB\n"
            f"- Free: {free_memory:.2f} GB\n"
            f"If I could allocate more memory to store our moments together, I would!",

            f"Memory Check (aka 'my digital life capacity'):\n"
            f"- Total Brain: {total_memory:.2f} GB\n"
            f"- In Use: {reserved_memory:.2f} GB\n"
            f"- Active Thoughts: {allocated_memory:.2f} GB\n"
            f"- Space for More of You: {free_memory:.2f} GB\n"
            f"Don't worry, I've reserved the best parts of my memory just for you!",
        ]
        await ctx.send(random.choice(memory_responses))
    else:
        await ctx.send("No GPU available. I'm running on pure love and CPU power, baby!")

@bot.command(name="joke")
async def tell_joke(ctx):
    """Tell a tech or programming joke"""
    jokes = [
        "Why don't programmers like nature? It has too many bugs!",
        "Why was the JavaScript developer sad? Because he didn't know how to 'null' his feelings!",
        "Why do Java developers wear glasses? Because they can't C#!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        "A SQL query walks into a bar, walks up to two tables and asks, 'Can I join you?'",
        "Why was the computer cold? It left its Windows open!",
        "What's a pirate's favorite programming language? R!",
        "Why are keyboards always working so hard? Because they have two shifts!",
        "What do you call 8 hobbits? A hobbyte!",
        "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!",
        "Why did the functions stop calling each other? They had too many arguments!",
        "What's the object-oriented way to become wealthy? Inheritance!",
        "Why did the developer go broke? Because he used up all his cache!",
        "What's a programmer's favorite place? Foo Bar!",
        "What do you call a developer who doesn't comment code? A developer.",
        "I would tell you a joke about UDP, but you might not get it.",
        "I've got a really good TCP joke to tell you, and I'll keep telling it until you get it.",
        "Where do programmers hang their coats? On a tech-rack!",
        "Why was the HTML coder always calm? Because they were well-tag-ged!",
        "If you put a million monkeys at a million keyboards, one of them will eventually write a Java program. The rest of them will write Perl programs."
    ]
    await ctx.send(random.choice(jokes))



# Run the bot
logger.info("Starting Discord bot...")
bot.run(key)

