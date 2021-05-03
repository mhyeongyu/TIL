```
sudo service Process_name start #Process 시작
sudo service Process_name stop #Process 중지
sudo service Process_name restart #Process 재시작
ps -ef | grep Keyword #이름에 Keyword를 포함하는 Process 검색


apt-get update #패키지 목록 업데이트
apt-get install Package #패키지 설치 (apt-get update 후 실행)
apt-get upgrade Package #패키지 업그레이드
apt-get remove Package #패키지 삭제
```

ex) apache2
```
sudo apt-get update #패키지 목록 업데이트
sudo apt-get install apache2 #아파치 설치
sudo service apache2 start #아파치 실행
ps -ef | grep apache2 #아파치 실행 확인
sudo service apache2 stop #아파치 종료


su - root계정 실행, 관리자권한으로 변경
Package --version #패키지 버전 확인
hostname -l #호스트 확인
```


## Linux
ls #디렉토리 목록 확인  
mkdir #새 디렉토리 생성  
cd #디렉토리 이동  
rm #디렉토리 삭제  
cp #파일 복사  
mv #파일 이동, 이름변경  
cat #파일 내용 보기  
find . *.log #확장자가 log인 파일 위치 검색  
pwd #현재 위치 확인  
clear #명령창 내용 삭제  
touch #파일 생성  


## AWS
#### EC2
-인스턴스  
퍼블릭 IPv4 주소  
퍼블릭 IPv4 DNS  

-보안그룹  
인바운드 규칙  
TCP유형  
포트범위, 소스  
SSH : 22  
HTTP : 80  
Mysql : 3306  

-로드밸런서  
포트구성  


#### RDS
-데이터베이스  
엔드포인트로 연결  

#### S3
simple storage service  
-버킷 생성  
웹서버 파일로드  