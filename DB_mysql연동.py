#!/usr/bin/env python
# coding: utf-8

# **mysql 연동하기**

# ### 필요한 모듈을 설치한다
# - pymysql 패키지를 설치한다
# - pip install pymysql 

# In[2]:


pip install pymysql


# 1. DB모듈 import하기
# 2. DB접속 : 접속객체 얻어오기
# 3. 쿼리작성하기
# 4. 쿼리 실행
# 5. 결과값을 얻어오기
# 6. DB접속 종료

# In[37]:


import pymysql

conn = pymysql.connect(
    host = 'localhost',
    port = 3306,   #방번호
    user = 'pbl',
    password = 'pblpw',
    db = 'pbldb'
)


# In[21]:


sql = 'select * from member'
cursor = conn.cursor()
cursor.execute(sql)


# In[9]:


for member in cursor:
    print(member)


# In[12]:


# cursor 재실행시 다시 객체를 받아와야함, 
# cursor는 포인터로 한번 이동하면 재실행으로 초기화 시켜주어야한다
# cursor: 포인터 이동 개념
result = cursor.fetchall() #한개 값만 반환 시 fetchone()
result


# In[22]:


result = cursor.fetchmany(2)
result


# In[16]:


result = cursor.fetchone()
result


# In[31]:


#딕셔너리 형태로 받아오기
dictCur = conn.cursor(pymysql.cursors.DictCursor)
print(dictCur.execute('select * from member'))
print()

dictResult = dictCur.fetchall()
print(dictResult)
print()

print(type(dictResult))


# In[52]:


d = dictResult[0]


# In[55]:


d['MEMBER_ID']


# **딕셔너리**
# - dict.keys()
# - dict.values()
# - dict.items() #튜플형태

# In[36]:


dictCur.close()
cursor.close()
conn.close()


# In[42]:


conn = pymysql.connect(
    host = 'localhost',
    port = 3306,   #방번호
    user = 'pbl',
    password = 'pblpw',
    db = 'pbldb'
)


# In[41]:


query = '''
insert member
values(null, 'choi@naver.com', '4567', 'choi', '22', default)
'''

cur = conn.cursor()
cur.execute(query)
conn.commit()
cursor.close()
conn.close()


# In[43]:


email = 'kim@naver.com'
password = '1234'
name = 'kim'

query = '''
insert member
values(null, %s, %s, %s, '22', default)
'''

cur = conn.cursor()
cur.execute(query, (email, password, name))
conn.commit()
cursor.close()
conn.close()

