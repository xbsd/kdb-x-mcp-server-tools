import logging
from typing import Dict, Any, Optional
from mcp_server.utils.kdbx import get_kdb_connection
from mcp_server.utils.embeddings_helpers import get_embedding_config

logger = logging.getLogger(__name__)

def _format_data(data, table=None) -> str:
    """Helper function to format data for display."""
    if table:
        embeddings_column, _, _, _, _ = get_embedding_config(table)
        if embeddings_column and hasattr(data, "pop"):
            data.pop(embeddings_column, None)
    if hasattr(data, 'to_string'):
        return data.to_string()
    return str(data)

async def describe_single_table_impl(table_name: str) -> Dict[str, Any]:
    """
    Implementation for describing a specific KDB table with metadata and sample rows.

    Args:
        table_name: Name of the KDB table to describe

    Returns:
        Dict[str, Any]: Table description with metadata and sample data
    """
    try:
        conn = get_kdb_connection()

        # Get table metadata
        total_records = conn('{count get x}', table_name).py()
        schema_data = conn.meta(table_name).py()
        partitioned_table = table_name in conn.Q.pt.py()

        # Format schema information
        schema_formatted = _format_data(schema_data)
        
        # Get preview data if table has records
        preview_data = None
        preview_formatted = None
        if total_records > 0:
            preview_size = min(3, total_records)
            if not partitioned_table:
                preview_data = conn('{x sublist get y}', preview_size, table_name).py()
            else:
                preview_data = conn('{.Q.ind[get y;til x]}', preview_size, table_name).py()
            preview_formatted = _format_data(preview_data, table_name)

        logger.info(f"Successfully analyzed table '{table_name}' with {total_records} records")
        
        return {
            "status": "success",
            "table_name": table_name,
            "total_records": total_records,
            "is_partitioned": partitioned_table,
            "schema": {
                "raw": schema_data,
                "formatted": schema_formatted
            },
            "preview": {
                "size": preview_size if total_records > 0 else 0,
                "data": preview_data,
                "formatted": preview_formatted
            } if total_records > 0 else None,
            "resource_uri": f"kdbx://tables/{table_name}",
            "description": f"KDB-X table analysis for '{table_name}'"
        }

    except Exception as e:
        logger.error(f"Failed to analyze table '{table_name}': {e}")
        return {
            "status": "error", 
            "table_name": table_name,
            "message": f"Failed to analyze table '{table_name}': {str(e)}"
        }

async def describe_all_tables_impl() -> Dict[str, Any]:
    """
    Implementation for describing all tables in the KDB database.

    Returns:
        Dict[str, Any]: Complete database schema overview
    """
    try:
        conn = get_kdb_connection()
        available_tables = conn.tables(None).py()

        if not available_tables:
            return {
                "status": "success",
                "message": "Database is empty - no tables found",
                "table_count": 0,
                "tables": [],
                "resource_uri": "kdbx://tables"
            }

        # Get detailed information for each table
        table_details = []
        for table_name in available_tables:
            table_info = await describe_single_table_impl(table_name)
            table_details.append(table_info)

        # Count successful vs failed analyses
        successful_tables = [t for t in table_details if t.get("status") == "success"]
        failed_tables = [t for t in table_details if t.get("status") == "error"]

        logger.info(f"Database analysis complete: {len(successful_tables)} successful, {len(failed_tables)} failed")
        
        return {
            "status": "success",
            "table_count": len(available_tables),
            "successful_analyses": len(successful_tables),
            "failed_analyses": len(failed_tables),
            "tables": table_details,
            "table_names": available_tables,
            "resource_uri": "kdbx://tables",
            "description": "Complete KDB-X database schema overview"
        }

    except Exception as e:
        logger.error(f"Database schema analysis failed: {e}")
        return {
            "status": "error", 
            "message": f"Database schema analysis failed: {str(e)}"
        }

def register_tools(mcp_server):
    """
    Register database table analysis tools with the MCP server.
    This function is called automatically during server startup.
    """
    
    @mcp_server.tool()
    async def get_kdbx_database_tables() -> Dict[str, Any]:
        """
        Retrieve comprehensive KDB-X database schema overview with all table details.
        
        This tool provides complete analysis of all tables in the KDB-X database,
        including schema information, record counts, partitioning status, and data previews.
        
        Features provided:
        - Complete list of all tables in the database
        - Schema information for each table (column names, types)
        - Record counts and partitioning status
        - Sample data preview (first 3 records) for non-empty tables
        - Embedding column handling (filtered out for readability)
        - Error handling for individual table analysis failures
        
        Use this tool when you need to:
        - Understand the database structure before writing queries
        - Get an overview of available data
        - Check table schemas and data types
        - See sample data to understand table contents
        - Plan SQL queries with proper table and column names
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: "success" or "error"
                - table_count: Number of tables found
                - successful_analyses: Number of successfully analyzed tables
                - failed_analyses: Number of failed table analyses  
                - tables: List of detailed table information
                - table_names: Simple list of table names
                - resource_uri: Original resource URI
                - description: Brief description of the analysis
                - message: Error message if status is "error"
        """
        return await describe_all_tables_impl()
    
    @mcp_server.tool()
    async def get_kdbx_table_info(table_name: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a specific KDB-X table.
        
        This tool provides comprehensive analysis of a single table, including
        schema details, record count, partitioning information, and sample data.
        
        Features provided:
        - Table schema with column names and data types
        - Total record count
        - Partitioning status (regular vs partitioned table)
        - Sample data preview (first 3 records)
        - Formatted output for easy reading
        - Error handling for invalid table names
        
        Use this tool when you need to:
        - Examine a specific table's structure before querying
        - Understand column names and data types
        - See sample data to understand the table's content
        - Check if a table is partitioned
        - Get detailed information for SQL query planning
        
        Args:
            table_name (str): Name of the KDB-X table to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: "success" or "error"
                - table_name: The analyzed table name
                - total_records: Number of records in the table
                - is_partitioned: Whether the table is partitioned
                - schema: Schema information (raw and formatted)
                - preview: Sample data preview (if table has records)
                - resource_uri: Resource URI for this specific table
                - description: Brief description of the table analysis
                - message: Error message if status is "error"
        """
        return await describe_single_table_impl(table_name)
    
    return ['get_kdbx_database_tables', 'get_kdbx_table_info']

