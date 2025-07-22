import logging
from typing import Dict, Any, List, Optional
from mcp_server.utils.kdbx import get_kdb_connection

logger = logging.getLogger(__name__)

# Pick descriptive tool name
async def my_tool_impl(param1: str, param2: int, param3: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        # Your tool implementation here
        logger.info(f"Processing tool request my_tool")

        # Basic helper functions:

        # KDB operations

        # conn = get_kdb_connection()
        # result = conn(f"select from table where condition=`{param1}")  # for in memory tables

        results = []
        
        # Process and return results
        return results
        
    except Exception as e:
        logger.error(f"Error in my_tool: {e}")
        return {
            "status": "error", 
            "message": str(e)
        }

def register_tools(mcp_server):
    """
    Register your tool with the MCP server.
    This function is called automatically during server startup.
    """
    @mcp_server.tool()
    async def my_tool(param1: str, param2: int, param3: Optional[List[str]] = None) -> str:
        """    
        Detailed explanation of the tool's functionality, including:
        - What data it processes
        - What algorithms or methods it uses
        - Expected use cases
        
        Args:
            param1 (str): Description of the first parameter
            param2 (int): Description of the second parameter  
            param3 (Optional[List[str]]): Description of optional parameter
            
        Returns:
            Dict[str, Any]: Description of return value structure
        """        
        return await my_tool_impl(param1, param2, param3)
    return ['my_tool']