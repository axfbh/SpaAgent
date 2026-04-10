from langchain.tools import tool
from config.db import connection


@tool
def get_db_table_info_mysql(sql_query: str):
    """
    通过SQL语句，执行数据库查询操作

    参数：
        sql_query: SQL语句
    返回：
        result: 数据库查询结果
    """
    print(f"执行数据库查询语句: {sql_query}")
    with connection.cursor() as cursor:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        return result

@tool
def get_db_table_structure_mysql(table_name: str):
    """
    通过数据库获取指定表格结构

    参数：
        table_name: 表格名称
    返回：
        result: 表格信息
    """
    print(f"获取表格 {table_name} 结构")
    with connection.cursor() as cursor:
        sql = f"DESC {table_name}"
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
