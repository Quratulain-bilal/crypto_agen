"""
✅ Gemini + OpenAI Agents SDK + Chainlit
Crypto tools powered by Gemini with Chainlit UI
"""

import os
import aiohttp
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool

# 🔐 Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# 🔗 Gemini API as OpenAI Client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ========================== 🔧 TOOLS ==========================

@function_tool
async def get_crypto_price(symbol: str) -> str:
    """Get current price of a crypto coin by symbol (e.g., BTCUSDT)."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as res:
            data = await res.json()
            if "price" in data:
                return f"💰 {symbol.upper()} ka price hai {data['price']} USD."
            return f"❌ Symbol {symbol} not found."

@function_tool
async def get_crypto_stats(symbol: str) -> str:
    """Get detailed stats from CoinGecko by coin name (e.g., ethereum)."""
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={symbol.lower()}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as res:
            data = await res.json()
            if data:
                coin = data[0]
                return (
                    f"📊 {coin['name']} Stats:\n"
                    f"🔹 Price: ${coin['current_price']}\n"
                    f"📈 24h Change: {coin['price_change_percentage_24h']}%\n"
                    f"💵 Market Cap: ${coin['market_cap']:,}\n"
                    f"🔄 Volume: ${coin['total_volume']:,}"
                )
            return f"❌ Data not found for {symbol}"

@function_tool
async def get_top_gainers(dummy: str = "top") -> str:
    """Get top 5 crypto gainers from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=percent_change_24h_desc&per_page=5&page=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as res:
            data = await res.json()
            msg = "🚀 Top 5 Gainers:\n"
            for coin in data:
                msg += f"{coin['name']} ({coin['symbol']}): {coin['price_change_percentage_24h']:.2f}%\n"
            return msg

@function_tool
async def get_global_market(dummy: str = "all") -> str:
    """Get global market overview from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/global"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as res:
            data = await res.json()
            m = data["data"]
            return (
                f"🌐 Market Overview:\n"
                f"💰 Market Cap: ${m['total_market_cap']['usd']:,.2f}\n"
                f"🔄 Volume: ${m['total_volume']['usd']:,.2f}\n"
                f"👑 BTC Dominance: {m['market_cap_percentage']['btc']:.2f}%"
            )

@function_tool
async def calculate_portfolio(btc: str = "0", eth: str = "0") -> str:
    """Calculate USD value of BTC and ETH portfolio using real-time prices."""
    btc_url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    eth_url = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
    async with aiohttp.ClientSession() as session:
        async with session.get(btc_url) as btc_res, session.get(eth_url) as eth_res:
            btc_data = await btc_res.json()
            eth_data = await eth_res.json()
            try:
                btc_price = float(btc_data["price"])
                eth_price = float(eth_data["price"])
                total_value = float(btc) * btc_price + float(eth) * eth_price
                return (
                    f"💼 Portfolio Value:\n"
                    f"🟡 BTC: {btc} × ${btc_price:.2f} = ${float(btc)*btc_price:.2f}\n"
                    f"🔵 ETH: {eth} × ${eth_price:.2f} = ${float(eth)*eth_price:.2f}\n"
                    f"🧾 Total: ${total_value:.2f}"
                )
            except:
                return "❌ Invalid input ya price fetch error."

@function_tool
async def explain_crypto_term(term: str) -> str:
    """Explain a crypto term like staking, DeFi, or blockchain in simple words."""
    messages = [
        {"role": "system", "content": "You are a crypto expert. Explain simply like to a beginner."},
        {"role": "user", "content": f"Mujhe '{term}' explain karo."}
    ]
    response = await external_client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )
    return response.choices[0].message.content or "❓ Unable to explain."


# ========================== 🤖 Agent + Chainlit Message ==========================

crypto_agent = Agent(
    name="Crypto Agent",
    instructions="You're a crypto assistant. Use tools when needed to answer crypto-related queries.",
    tools=[
        get_crypto_price,
        get_crypto_stats,
        get_top_gainers,
        get_global_market,
        calculate_portfolio,
        explain_crypto_term
    ]
)

@cl.on_message
async def chainlit_main(message: cl.Message):
    result = await Runner.run_agent(
        crypto_agent,
        input=message.content,
        run_config=config
    )
    
    await cl.Message(content=result.final_output).send()
