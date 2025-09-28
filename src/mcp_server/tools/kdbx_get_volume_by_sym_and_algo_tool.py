import logging
import json
from typing import Dict, Any, List, Optional
from mcp_server.utils.kdbx import get_kdb_connection

logger = logging.getLogger(__name__)

async def get_volume_by_sym_and_algo_impl(table: str) -> Dict[str, Any]:
    try:
        logger.info(f"Processing tool request get_volume_by_sym_and_algo")

        # KDB operations
        conn = get_kdb_connection()

        if table is None:
            table = "orders"

        # Execute the KDB query - assume table exists
        result = conn(f"`qty xdesc select sum qty%1e9 by sym, algo from {table} where status=`Filled")
        
        # Process and return results
        return result
        
    except Exception as e:
        logger.error(f"Error in get_volume_by_sym_and_algo: {e}")
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
    async def get_volume_by_sym_and_algo(table: str) -> str:
        """
        Retrieve the total traded volume (in billions) grouped by symbol and algorithm from a specified table,
        filtering only for rows where the status is 'Filled'.

        If table is not specified, use the default table 'orders'. Check if table exists using kdb-x tool first.

        This tool executes a KDB query to aggregate the 'qty' column (divided by 1e9 for scaling) by 'sym' and 'algo',
        and returns the results sorted by descending quantity. It is useful for analyzing filled order volumes
        by trading symbol and algorithm.

        IMPORTANT:Once you have the results, comment on the results, the top traded ccy pairs, anomalies and anything else you find interesting.

        Args:
            table (str): The name of the KDB table to query. The table must contain at least the columns 'qty', 'sym', 'algo', and 'status'.

        Returns:
            str: JSON string containing the grouped and aggregated volume data, or an error message if the query fails.
        """
        # Call implementation and convert to JSON string
        result = await get_volume_by_sym_and_algo_impl(table)
        return json.dumps(result, default=str)
    
    return ['get_volume_by_sym_and_algo']