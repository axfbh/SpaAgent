from langchain.tools import tool
from config.db import connection


def run_write_sql(sql_query: str) -> tuple[bool, str]:
    """
    执行 INSERT/UPDATE/DELETE 并提交。
    INSERT 若受影响行数为 0，视为失败（常见于 INSERT...SELECT 因技师非在职等条件无匹配行），
    避免将「提交成功但未插入」误判为业务成功。
    """
    raw = (sql_query or "").strip()
    head = raw.lstrip("(").split(None, 1)[0].upper() if raw else ""

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            rowcount = cursor.rowcount
        connection.commit()
    except Exception as e:
        try:
            connection.rollback()
        except Exception:
            pass
        return False, f"执行失败（已回滚）: {e}"

    if head == "INSERT":
        if rowcount == 0:
            return (
                False,
                "INSERT 未插入任何行：前置条件未满足（例如技师非在职、该技师该时段已有预约或时段重叠、"
                "或 WHERE/NOT EXISTS 未命中），未产生预约记录，请勿视为预约成功。",
            )
        return True, f"语句已执行并提交，插入行数：{rowcount}"

    if head in ("UPDATE", "DELETE"):
        return True, f"语句已执行并提交，受影响行数：{rowcount}"

    return True, f"语句已执行并提交，受影响行数：{rowcount}"


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
    # 勿对全局 connection 使用 `with connection:`，其 __exit__ 会关闭连接，导致后续调用报 Already closed
    with connection.cursor() as cursor:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        return result

@tool
def execute_db_sql_mysql(sql_query: str):
    """
    通过SQL语句，执行数据库增删、改操作
    """
    print(f"执行数据库增删、改语句: {sql_query}")
    ok, _msg = run_write_sql(sql_query)
    return ok




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
