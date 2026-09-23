#!/usr/bin/env python3
"""
IO Gateway Service
Responsible for:
1. Connecting to OKX WebSocket
2. Receiving real-time market data
3. Persisting data to the WAL (Write-Ahead Log) — durability first
4. Publishing data to shared memory ring buffers for the compute layer
"""

import asyncio
import signal
import sys
# import psutil
from loguru import logger
from typing import Set

from io_gateway.ws_client import OKXWebSocketClient

class IOGatewayService:
    def __init__(self):
        self.ws_client = OKXWebSocketClient()
        self.running = False
        # self._setup_cpu_affinity()
        
    def _setup_cpu_affinity(self):
        pass
        # """Bind process to Efficiency Cores (E-cores)"""
        # try:
        #     # On M3 chips, E-cores are usually the last cores
        #     # This is a simple heuristic, might need adjustment
        #     cpu_count = psutil.cpu_count(logical=False)
        #     if cpu_count >= 8:
        #         p = psutil.Process()
        #         # Use last 4 cores (typically E-cores on Apple Silicon)
        #         e_cores = list(range(cpu_count - 4, cpu_count))
        #         if hasattr(p, 'cpu_affinity'):
        #             p.cpu_affinity(e_cores)
        #             logger.info(f"Bound IO Gateway to cores: {e_cores}")
        # except Exception as e:
        #     logger.warning(f"Failed to set CPU affinity: {e}")

    async def start(self):
        """Start the service"""
        self.running = True
        logger.info("Starting IO Gateway Service...")
        
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))
            
        try:
            await self.ws_client.start()
        except Exception as e:
            logger.error(f"IO Gateway service error: {e}")
            await self.shutdown()

    async def shutdown(self, sig=None):
        """Shutdown the service"""
        if not self.running:
            return
            
        if sig:
            logger.info(f"Received signal {sig.name}, shutting down...")
            
        self.running = False
        await self.ws_client.stop()
        logger.info("IO Gateway Service stopped")
        
        # Cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
            
        await asyncio.gather(*tasks, return_exceptions=True)
        sys.exit(0)

def main():
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )
    
    service = IOGatewayService()
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
