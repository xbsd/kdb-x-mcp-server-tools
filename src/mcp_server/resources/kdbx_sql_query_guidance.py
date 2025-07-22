
import logging

logger = logging.getLogger(__name__)

def kdbx_sql_query_guidance_impl() -> str:
    path = "src/mcp_server/resources/kdbx_sql_query_guidance.txt"
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def register_resources(mcp_server):
    @mcp_server.resource("file://guidance/kdbx-sql-queries")
    async def kdbx_sql_query_guidance() -> str:
        """
        Provides guidance when using SQL select statements with the kdbx_run_sql_query tool.

        Returns:
            str: Details and examples on supported select statement when using the sql tool.
        """
        return kdbx_sql_query_guidance_impl()
    return ['file://guidance/kdbx-sql-queries']