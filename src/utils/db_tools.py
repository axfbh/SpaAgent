from langchain.tools import tool
from config.db import connection


@tool
def get_therapist_info_mysql(therapist_id: int=None):
    """
    通过数据库therapists表，获取指定技师的信息，如果未指定技师工号，则获取所有技师的信息

    参数：
        therapist_id: 技师工号，如果为None，则获取所有技师的信息
    返回：
        result: 技师信息
    """
    print(f"获取技师信息，技师工号: {therapist_id}")
    with connection.cursor() as cursor:
        if therapist_id:
            sql = "SELECT * FROM `therapists` WHERE `therapist_id` = %s"
            cursor.execute(sql, (therapist_id,))
        else:
            sql = "SELECT * FROM `therapists`"
            cursor.execute(sql)
        result = cursor.fetchall()
        return result


def get_db_table_info(table_name: str):
    """
    通过数据库获取指定表格结构信息

    参数：
        table_name: 表格名称
    返回：
        result: 表格信息
    """
    print(f"获取表格信息，表格名称: {table_name}")
    with connection.cursor() as cursor:
        sql = f"DESC {table_name}"
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
