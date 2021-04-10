package sec02.exam01;

public class SignOperatorExample {

	public static void main(String[] args) {
		int x = -100;
		int result1 = +x;
		int result2 = -x;
		System.out.println(result1);
		System.out.println(result2);
		
		byte b = 100;
		//byte result3 = -b;   부호 연산의 결과는 int 타입이므로 int 타입 변수에 저장해야한다.
		int result3 = -b;
		System.out.println(result3);
		

	}

}
