import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='250309',
                             database='test_spa',
                             cursorclass=pymysql.cursors.DictCursor)

# with connection:
#     with connection.cursor() as cursor:
#         sql = "SELECT * FROM `therapists`"
#         cursor.execute(sql)
#         result = cursor.fetchall()
#         # print(result)