"""Spot prices for BTC and ETH read straight off Uniswap v3 pools on Base,
bypassing every off-chain price API. A pool's own state (`slot0`) already is
the price: the number an on-chain trade against that pool would get right
now, in USDC per coin.

No ABI library is used. The handful of functions this needs (`token0`,
`token1`, `fee`, `liquidity`, `slot0`, and the factory's `getPool`) are
called by their 4-byte selectors -- the first 4 bytes of
keccak256("name(types)"), fixed forever by the ABI spec and identical on
every deployment, so they're written below as constants rather than computed
with a keccak library this project doesn't have. Their fixed-width return
values are then cut out of the result hex by hand.

Pool addresses are not trusted from memory or the web. Each entry in POOLS
was checked on 2026-09-05 by calling token0()/token1()/fee() on the pool and
confirming the tokens are the real Base WETH/USDC/cbBTC (addresses below),
and by calling the factory's getPool() across all four fee tiers for
cbBTC/USDC and comparing liquidity() to find the deepest one:
  fee tier      liquidity (raw)
  0.01%  (100)   6.4e7
  0.05%  (500)   2.06e12   <- deepest, used below
  0.30% (3000)   4.48e11
  1.00% (10000)  5.58e8
WETH/USDC only has meaningful depth at 0.05%, which is also the address
given as a starting candidate; it verified clean.
"""
from __future__ import annotations

import httpx

# Tried in order; a dead or rate-limited endpoint just falls through to the
# next one. base.llamarpc.com returned HTTP 521 (host down) during testing
# but is kept as a last resort in case that was transient.
RPCS = [
    "https://mainnet.base.org",
    "https://base-rpc.publicnode.com",
    "https://1rpc.io/base",
    "https://base.llamarpc.com",
]

_SLOT0 = "0x3850c7bd"      # slot0() -> sqrtPriceX96 is the first 32-byte word
_TOKEN0 = "0x0dfe1681"     # token0() -> address
_TOKEN1 = "0xd21220a7"     # token1() -> address
_FEE = "0xddca3f43"        # fee() -> uint24
_LIQUIDITY = "0x1a686502"  # liquidity() -> uint128
_GET_POOL = "0x1698ee82"   # factory.getPool(address,address,uint24) -> address

WETH = "0x4200000000000000000000000000000000000006"
USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
CBBTC = "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf"
FACTORY = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD"

# market -> pool address, its fee tier, the two tokens' decimals, and whether
# the "base" coin (BTC or ETH) is token0 or token1 -- Uniswap orders a pool's
# two tokens by address, not by which one is conventionally the base.
POOLS = {
    "btcusd": {"address": "0xfbb6eed8e7aa03b138556eedaf5d271a5e1e43ef",
               "fee_bp": 5, "base_decimals": 8, "quote_decimals": 6, "base_is_token0": False},
    "ethusd": {"address": "0xd0b53d9277642d899df5c87a3966a349a798f224",
               "fee_bp": 5, "base_decimals": 18, "quote_decimals": 6, "base_is_token0": True},
}


def _post(rpc: str, payload: list[dict], timeout: float = 8.0) -> list:
    """Send a JSON-RPC batch to one endpoint; raise on any transport or
    RPC-level error. Returns results in the same order as `payload`."""
    r = httpx.post(rpc, json=payload, timeout=timeout)
    r.raise_for_status()
    replies = {rep["id"]: rep for rep in r.json()}
    out = []
    for call in payload:
        rep = replies[call["id"]]
        if "error" in rep:
            raise RuntimeError(f"{rpc} {call['method']}: {rep['error']}")
        out.append(rep["result"])
    return out


def _call_rpcs(payload: list[dict]) -> list:
    """Try each RPC in RPCS in order; raise the last failure if all fail."""
    err: Exception | None = None
    for rpc in RPCS:
        try:
            return _post(rpc, payload)
        except Exception as e:  # network error, timeout, rate limit, bad JSON
            err = e
    raise RuntimeError(f"all RPCs failed, last error: {err}")


def block_number() -> int:
    (result,) = _call_rpcs([{"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}])
    return int(result, 16)


def quote(market: str) -> tuple[float, float, float, float, int]:
    """(bid, ask, bid size, ask size, block) in the shape of an exchange
    quote, so the pool can sit in the tick file as one more venue. The pool
    has a single price; selling into it or buying from it costs the fee
    tier each way, so that is the "spread". Size is the coins you can trade
    before moving the pool's price by one basis point -- from the pool's
    current liquidity, which is what a real trade would consume. Reads
    slot0, liquidity and the block number in one batch from the same RPC
    so all three describe the same moment; raises RuntimeError if every RPC
    in RPCS fails."""
    pool = POOLS[market]
    payload = [
        {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": pool["address"], "data": _SLOT0}, "latest"]},
        {"jsonrpc": "2.0", "id": 2, "method": "eth_call", "params": [{"to": pool["address"], "data": _LIQUIDITY}, "latest"]},
        {"jsonrpc": "2.0", "id": 3, "method": "eth_blockNumber", "params": []},
    ]
    slot0_hex, liq_hex, block_hex = _call_rpcs(payload)
    sqrt_p = int(slot0_hex[2:66], 16) / 2**96  # sqrt of raw token1 units per raw token0 unit
    liquidity = int(liq_hex, 16)
    raw = sqrt_p ** 2
    scale = 10 ** (pool["base_decimals"] - pool["quote_decimals"])
    # base is token0: raw already converts token0 (base) into token1 (quote).
    # base is token1: raw converts the other way, so invert it.
    p = raw * scale if pool["base_is_token0"] else scale / raw
    # Uniswap v3: token1 moved = liquidity * change in sqrt(price); a 1 bp price move is a 0.5 bp move in its root
    token1_for_1bp = liquidity * sqrt_p * 0.5e-4
    if pool["base_is_token0"]:
        coins = token1_for_1bp / 10 ** pool["quote_decimals"] / p  # token1 is the quote: dollars, so divide by price
    else:
        coins = token1_for_1bp / 10 ** pool["base_decimals"]  # token1 is the coin itself
    f = pool["fee_bp"] / 1e4
    return p * (1 - f), p * (1 + f), coins, coins, int(block_hex, 16)


def price(market: str) -> tuple[float, int]:
    """(price in USDC per coin, block number the price was read at)."""
    bid, ask, _, _, block = quote(market)
    return (bid + ask) / 2, block
