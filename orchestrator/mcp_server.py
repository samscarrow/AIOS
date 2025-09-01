#!/usr/bin/env python3
"""
MCP Server for AIOS Strategic Orchestration
Implements the Model Context Protocol for LMStudio integration
"""

import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional

# Import our orchestrator
from aios_strategic_orchestrator import AIOSStrategicOrchestrator, AIOS_ORCHESTRATION_TOOLS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    """MCP Server implementation for AIOS Strategic Orchestration"""
    
    def __init__(self):
        self.orchestrator = AIOSStrategicOrchestrator()
        self.request_id = 0
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return await self.handle_initialize(request_id, params)
            elif method == "tools/list":
                return await self.handle_tools_list(request_id)
            elif method == "tools/call":
                return await self.handle_tool_call(request_id, params)
            else:
                return self.error_response(request_id, -32601, f"Method not found: {method}")
                
        except Exception as e:
            logger.error(f"Error handling request {method}: {e}")
            return self.error_response(request_id, -32603, str(e))
    
    async def handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    }
                },
                "serverInfo": {
                    "name": "aios-strategic-orchestration",
                    "version": "1.0.0"
                }
            }
        }
    
    async def handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools/list request"""
        tools = []
        for tool_config in AIOS_ORCHESTRATION_TOOLS:
            tools.append({
                "name": tool_config["name"],
                "description": tool_config["description"],
                "inputSchema": tool_config["parameters"]
            })
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools
            }
        }
    
    async def handle_tool_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "initiate_thought_stream":
                result = await self.orchestrator.initiate_thought_stream(
                    task_description=arguments["task_description"],
                    agent_name=arguments["agent_name"],
                    cognitive_type=arguments.get("cognitive_type", "reasoning"),
                    complexity=arguments.get("complexity", "medium"),
                    semantic_tags=arguments.get("semantic_tags", [])
                )
            elif tool_name == "contribute_thought":
                result = await self.orchestrator.contribute_thought(
                    stream_id=arguments["stream_id"],
                    agent_name=arguments["agent_name"],
                    thought_content=arguments["thought_content"],
                    cognitive_action=arguments.get("cognitive_action", "analysis"),
                    attention_request=arguments.get("attention_request", 0.5)
                )
            elif tool_name == "execute_strategic_orchestration":
                result = await self.orchestrator.execute_strategic_orchestration(
                    stream_id=arguments["stream_id"],
                    agent_name=arguments["agent_name"],
                    execution_prompt=arguments["execution_prompt"],
                    task_type=arguments.get("task_type", "reasoning")
                )
            elif tool_name == "search_thought_streams":
                result = await self.orchestrator.search_thought_streams(
                    agent_name=arguments["agent_name"],
                    semantic_query=arguments.get("semantic_query"),
                    cognitive_type=arguments.get("cognitive_type"),
                    cognitive_state=arguments.get("cognitive_state"),
                    attention_threshold=arguments.get("attention_threshold"),
                    limit=arguments.get("limit", 10)
                )
            elif tool_name == "get_aios_tutorial":
                result = await self.orchestrator.get_aios_tutorial(
                    tutorial_type=arguments["tutorial_type"],
                    agent_name=arguments.get("agent_name", "NewAgent")
                )
            else:
                return self.error_response(request_id, -32601, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Tool call error for {tool_name}: {e}")
            return self.error_response(request_id, -32603, f"Tool execution failed: {str(e)}")
    
    def error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

async def main():
    """Main MCP server loop"""
    server = MCPServer()
    logger.info("AIOS Strategic Orchestration MCP Server starting...")
    
    try:
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSON request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    continue
                
                # Handle request
                response = await server.handle_request(request)
                
                # Send response to stdout
                print(json.dumps(response), flush=True)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)
    
    logger.info("AIOS Strategic Orchestration MCP Server shutting down")

if __name__ == "__main__":
    asyncio.run(main())