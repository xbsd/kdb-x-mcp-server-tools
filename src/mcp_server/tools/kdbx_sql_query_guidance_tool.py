import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

async def get_kdbx_sql_query_guidance_impl() -> Dict[str, Any]:
    """
    Implementation for retrieving KDB-X SQL query guidance content.
    This replaces the MCP resource functionality for Open WebUI compatibility.
    
    Returns:
        Dict[str, Any]: The guidance content and metadata
    """
    try:
        # Get the absolute path to the resource file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resource_path = os.path.join(
            current_dir, 
            "..", 
            "resources", 
            "kdbx_sql_query_guidance.txt"
        )
        
        # Normalize the path
        resource_path = os.path.normpath(resource_path)
        
        logger.info(f"Attempting to read guidance file from: {resource_path}")
        
        if not os.path.exists(resource_path):
            logger.error(f"Guidance file not found at: {resource_path}")
            return {
                "status": "error",
                "message": f"Guidance file not found at: {resource_path}"
            }
        
        with open(resource_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        logger.info("Successfully retrieved KDB-X SQL query guidance")
        
        return {
            "status": "success",
            "resource_uri": "file://guidance/kdbx-sql-queries",
            "content_type": "text/plain",
            "content": content,
            "description": "KDB-X SQL query guidance and examples"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving KDB-X SQL query guidance: {e}")
        return {
            "status": "error", 
            "message": str(e)
        }


def register_tools(mcp_server):
    """
    Register the KDB-X SQL query guidance tool with the MCP server.
    This function is called automatically during server startup.
    """
    @mcp_server.tool()
    async def get_kdbx_sql_query_guidance() -> Dict[str, Any]:
        """
        Retrieve KDB-X SQL query guidance and examples.
        
        This tool provides comprehensive guidance for writing SQL queries against KDB-X databases,
        including supported syntax, features, limitations, and practical examples. The guidance
        covers ANSI-compliant SQL operations, KDB-specific differences, and common use cases.
        
        Features covered:
        - Basic SELECT operations with DISTINCT, AS aliases
        - Aggregate functions: SUM, AVG, COUNT, MIN, MAX, FIRST, LAST, TOTAL
        - JOIN operations: LEFT, RIGHT, INNER, CROSS joins
        - Filtering with WHERE clauses, LIKE patterns, subqueries
        - GROUP BY and HAVING clauses
        - Query combination: UNION, INTERSECT, EXCEPT
        - Common Table Expressions (CTEs)
        - Data type handling and casting
        - Date/time operations
        - Pattern matching examples
        
        Use this guidance when:
        - Writing SQL queries for KDB-X databases
        - Understanding KDB-specific SQL syntax differences
        - Learning supported operations and limitations
        - Finding examples for common query patterns
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: "success" or "error"
                - content: Full guidance text content
                - resource_uri: Original resource URI
                - content_type: MIME type of content
                - description: Brief description of the resource
                - message: Error message if status is "error"
        """
        return await get_kdbx_sql_query_guidance_impl()
    
    return ['get_kdbx_sql_query_guidance']
