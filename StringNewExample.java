package chap05;

public class StringNewExample {

	public static void main(String[] args) {
		String strVar1 = "강호동";
		String strVar2 = "강호동";
		
		if (strVar1 == strVar2) {
			System.out.println("참조 true");
		} else {
			System.out.println("참조 false");
		}
		if (strVar1.equals(strVar2)) {
			System.out.println("문자열 true");
		} else {
			System.out.println("문자열 false");
		}
		
		String strVar3 = new String("강호동");
		String strVar4 = new String("강호동");
		
		if (strVar3 == strVar4) {
			System.out.println("참조 true");
		} else {
			System.out.println("참조 false");
		}
		if (strVar3.equals(strVar4)) {
			System.out.println("문자열 true");
		} else {
			System.out.println("문자열 false");
		}
	}

}
