{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pymysql in c:\\users\\admin\\anaconda3\\lib\\site-packages (1.0.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pymysql"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pymysql\n",
    "import pandas as pd\n",
    "import re\n",
    "from datetime import datetime, timedelta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#DB 접속\n",
    "def connection():\n",
    "    conn = pymysql.connect(\n",
    "    host = 'localhost',    \n",
    "    port = 3306,\n",
    "    user = 'root',\n",
    "    passwd = 'root&*',\n",
    "    db = 'zoomdb'\n",
    "    )\n",
    "    return conn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "#DB 접속 해제\n",
    "def close(input_conn):\n",
    "    input_conn.commit()\n",
    "    input_conn.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#0. 메인 메뉴\n",
    "def menu():\n",
    "    print('*' * 72)\n",
    "    print(' 1. 회원가입 | 2. 로그인 | 3. zoom 날짜 등록 | 4. 스터디 모임 찾기 | 5. 게시판 ')\n",
    "    print(' 프로그램 종료: q')\n",
    "    choice = input(' 선택: ')\n",
    "\n",
    "    return choice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#0-1. 게시판 메뉴\n",
    "def board_menu():\n",
    "    print(' 1. QnA 등록  |  2. QnA 조회  |  3. 익명글 등록  |  4. 익명 게시판 조회 ')\n",
    "    board_choice = input(' 선택: ')\n",
    "    return board_choice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#1. 회원가입\n",
    "def join():\n",
    "    conn = connection()\n",
    "    print('<<< 회원가입 >>>')\n",
    "    final_email = check_email()\n",
    "    final_passwd = check_passwd()\n",
    "    final_name = check_name()\n",
    "    st_group = input(\"스터디조: \")\n",
    "    final_ph = check_ph()\n",
    "\n",
    "    query = '''insert into student values(null, %s, %s, %s, %s, %s)'''\n",
    "    cur = conn.cursor()\n",
    "    cur.execute(query, (final_email, final_passwd, final_name, st_group, final_ph))\n",
    "    cur.close()\n",
    "    \n",
    "    close(conn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#1-1. 이메일 유효성 검사\n",
    "def check_email():\n",
    "    while True:\n",
    "        st_email = input(\"이메일 주소: \")\n",
    "        valid_email = re.match('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$', st_email)\n",
    "\n",
    "        if not valid_email:\n",
    "            print('유효하지 않은 이메일 주소입니다.')\n",
    "            print('다시 작성해 주세요.')\n",
    "        else: break\n",
    "    \n",
    "    return st_email"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#1-2. 비밀번호 유효성 검사\n",
    "def check_passwd():\n",
    "    while True:\n",
    "        st_passwd = input(\"비밀번호: \")\n",
    "        if st_passwd == '': print('비밀번호는 필수 입력 입니다.')\n",
    "        elif len(st_passwd) < 4: print('4자리 이상 입력해 주세요.')\n",
    "        else: break\n",
    "    return st_passwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#1-3. 이름 유효성 검사\n",
    "def check_name():\n",
    "    while True:\n",
    "        st_name = input('이름: ')\n",
    "        if st_name == '': print('이름은 필수 입력 입니다.')\n",
    "        else: break\n",
    "    return st_name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#1-4. 휴대전화번호 유효성 검사\n",
    "def check_ph():\n",
    "    while True:\n",
    "        st_ph = input(\"연락처(- 제외 11자리):  \")\n",
    "        ph1 = st_ph[0:3]\n",
    "        ph2 = st_ph[3:7]\n",
    "        ph3 = st_ph[7:]\n",
    "\n",
    "        if ph1 == '010' or ph1 == '011':\n",
    "            if len(ph2) < 5 and re.match('[0-9]', ph2):\n",
    "                if len(ph3) < 5 and re.match('[0-9]', ph3): break\n",
    "                else: \n",
    "                    print('유효하지 않은 연락처입니다.')\n",
    "                    print('다시 작성해 주세요.')\n",
    "                    continue\n",
    "            else: \n",
    "                print('유효하지 않은 연락처입니다.')\n",
    "                print('다시 작성해 주세요.')\n",
    "                continue\n",
    "        else:\n",
    "            print('유효하지 않은 연락처입니다.')\n",
    "            print('다시 작성해 주세요.')\n",
    "            continue\n",
    "\n",
    "    return st_ph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#2. 로그인\n",
    "def signin():\n",
    "    while True:\n",
    "        conn = connection()\n",
    "        print('*' * 32, 'log in', '*' * 32)\n",
    "        login_email = input('이메일 주소 입력: ')\n",
    "        login_pw = input('비밀번호 입력: ')\n",
    "\n",
    "        query = '''\n",
    "        select st_id from student \n",
    "        where st_email = %s and st_passwd = %s\n",
    "        '''\n",
    "        cur = conn.cursor(pymysql.cursors.DictCursor)\n",
    "        cur.execute(query, (login_email, login_pw))\n",
    "        result = cur.fetchone()\n",
    "        \n",
    "        if result is None: print('이메일 주소 또는 비밀번호가 잘못 입력되었습니다.')\n",
    "        else: \n",
    "            st_id = result['st_id']\n",
    "            print('로그인 완료')\n",
    "            break\n",
    "    cur.close()\n",
    "    close(conn)\n",
    "    return st_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#3. zoom 날짜 등록\n",
    "def zoom_date(id_param):\n",
    "    conn = connection()\n",
    "\n",
    "    t = datetime.today().strftime(\"%Y-%m-%d\") \n",
    "    w = (datetime.today() + timedelta(days = 5)).strftime(\"%Y-%m-%d\")\n",
    "    print('날짜:', t, '~', w)\n",
    "    print('zoom 참석일과 스터디 희망여부를 적어주세요.')\n",
    "\n",
    "    zoom_date_dict = {}\n",
    "    cnt = 0\n",
    "    while cnt < 3:\n",
    "        cnt += 1\n",
    "        zoom_date = input(str(cnt) + '번째 zoom_date 날짜를 입력하세요. ex)01-01 ')\n",
    "        attend = input('참석여부를 입력하세요. (y/n) ')\n",
    "        zoom_date_dict [zoom_date] = attend\n",
    "\n",
    "    for key, value in zoom_date_dict.items():\n",
    "        query = '''insert into zoom_list(st_id, zoom_date, study) \n",
    "                   values(%s, %s, %s)'''\n",
    "        cur = conn.cursor()\n",
    "        cur.execute(query, (id_param, key, value))\n",
    "\n",
    "    cur.close()\n",
    "    close(conn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#4. 날짜 선택\n",
    "def date_choice():\n",
    "    conn = connection()\n",
    "    today = datetime.today() + timedelta(days = 6) \n",
    "    t_lst = []\n",
    "    for i in range(5):\n",
    "        t_lst.append(today.strftime(\"%Y-%m-%d\"))\n",
    "        today += timedelta(days = 1)\n",
    "        \n",
    "    print('번호를 선택하세요.')\n",
    "    for idx, day in enumerate(t_lst):\n",
    "        print(idx + 1, '번: ', day)\n",
    "    want = int(input())\n",
    "\n",
    "    if want == 1: day = t_lst[0][5:]\n",
    "    elif want == 2: day = t_lst[1][5:]\n",
    "    elif want == 3: day = t_lst[2][5:]\n",
    "    elif want == 4: day = t_lst[3][5:]\n",
    "    else: day = t_lst[4][5:]\n",
    "    print('*' * 20, day, 'zoom으로 수업 듣는 명단', '*' * 20)\n",
    "\n",
    "    sql = '''\n",
    "        select st_name 이름, st_email 이메일, st_ph 연락처, st_group 조 from student s\n",
    "        join zoom_list z on s.st_id = z.st_id\n",
    "        where z.study = 'y' and z.zoom_date = %s\n",
    "        '''\n",
    "    dictCur = conn.cursor(pymysql.cursors.DictCursor)\n",
    "    dictCur.execute(sql, (day))\n",
    "    dictResult = dictCur.fetchall()\n",
    "    dictCur.close()\n",
    "    close(conn)\n",
    "    \n",
    "    df = pd.DataFrame(dictResult)\n",
    "    if df.shape == (0, 0): print('no data')\n",
    "    else: print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#5-1. Qna 게시판 등록\n",
    "def qna(id_param):\n",
    "    conn = connection()\n",
    "    print('*'*30, 'Q&A 게시판', '*'*30)\n",
    "    title = input('글의 제목을 입력하세요. ')\n",
    "    contents = input('글의 내용을 입력하세요. ')\n",
    "    \n",
    "    query = '''insert into qna values(null, %s, %s, %s, default)'''\n",
    "    cur = conn.cursor()\n",
    "    cur.execute(query, (id_param, title, contents))\n",
    "    cur.close()\n",
    "    close(conn)\n",
    "    \n",
    "    return print('글 등록이 완료되었습니다.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#5-2. Qna 조회\n",
    "def qna_view():\n",
    "    conn = connection()\n",
    "    print('*'*30, 'Q&A 게시판', '*'*30)\n",
    "    query = '''select st_name 작성자, title 제목, contents 내용, \n",
    "               date_format(written, '%m-%d') 작성일 from qna''' \n",
    "    dictCur = conn.cursor(pymysql.cursors.DictCursor)\n",
    "    dictCur.execute(query)\n",
    "    dictResult = dictCur.fetchall()\n",
    "    dictCur.close()\n",
    "    close(conn)\n",
    "\n",
    "    df = pd.DataFrame(dictResult)\n",
    "    print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#5-3. 익명 게시판 등록\n",
    "def add_descript():\n",
    "    conn = connection()\n",
    "    title = input('제목: ')\n",
    "    descript = input('내용: ')\n",
    "    sql = '''\n",
    "          insert no_name_board\n",
    "          values(null, default, %s, %s)\n",
    "          '''\n",
    "    cur = conn.cursor()\n",
    "    cur.execute(sql, (title, descript))\n",
    "    cur.close()\n",
    "    close(conn)\n",
    "    \n",
    "    return print('입력이 완료되었습니다.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#5-4. 익명 게시판 조회\n",
    "def print_board():\n",
    "    conn = connection()\n",
    "    sql = '''\n",
    "            select title 제목, descript 내용, date_format(write_day, '%m-%d') 작성일\n",
    "            from no_name_board\n",
    "          '''\n",
    "    dictCur = conn.cursor(pymysql.cursors.DictCursor)\n",
    "    dictCur.execute(sql)\n",
    "    dictResult = dictCur.fetchall()\n",
    "    dictCur.close()\n",
    "    close(conn)\n",
    "    \n",
    "    df = pd.DataFrame(dictResult)\n",
    "    print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "************************************************************************\n",
      " 1. 회원가입 | 2. 로그인 | 3. zoom 날짜 등록 | 4. 스터디 모임 찾기 | 5. 게시판 \n",
      " 프로그램 종료: q\n",
      "\n",
      " 선택: 4\n",
      "\n",
      "번호를 선택하세요.\n",
      "1 번:  2021-04-19\n",
      "2 번:  2021-04-20\n",
      "3 번:  2021-04-21\n",
      "4 번:  2021-04-22\n",
      "5 번:  2021-04-23\n",
      "1\n",
      "\n",
      "******************** 04-19 zoom으로 수업 듣는 명단 ********************\n",
      "    이름                   이메일          연락처   조\n",
      "0  문현규  gusrb4805@icloud.com  01011115555  1조\n",
      "1  문현규  gusrb4805@icloud.com  01011115555  1조\n",
      "2  문현규  gusrb4805@icloud.com  01011115555  1조\n",
      "3  배예진   byjin0229@gmail.com  01037845210  1조\n",
      "4  김강산    ka12sa3@icloud.com  01037894561  5조\n",
      "5  권소희    sohe0214@naver.com  01012223434  3조\n",
      "\n",
      "\n",
      "************************************************************************\n",
      " 1. 회원가입 | 2. 로그인 | 3. zoom 날짜 등록 | 4. 스터디 모임 찾기 | 5. 게시판 \n",
      " 프로그램 종료: q\n",
      "\n",
      " 선택: q\n",
      "\n"
     ]
    }
   ],
   "source": [
    "while True:\n",
    "    choice = menu()\n",
    "    if choice == 'q': break\n",
    "    elif choice == '1': join()\n",
    "    elif choice == '2': st_id_output = signin()\n",
    "    elif choice == '3': zoom_date(st_id_output)\n",
    "    elif choice == '4':\n",
    "        date_choice()\n",
    "    else: \n",
    "        input_choice = board_menu()\n",
    "        if input_choice == '1': qna(st_id_output)\n",
    "        elif input_choice == '2': qna_view()\n",
    "        elif input_choice == '3': add_descript()\n",
    "        elif input_choice == '4': print_board()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
