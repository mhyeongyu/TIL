package chap04;

public class SwitchExample {

	public static void main(String[] args) {
		int num = (int) (Math.random() * 6) +1;
		
		switch(num) {
			case 1: System.out.println("num: 1"); break;
			case 2: System.out.println("num: 2"); break;
			case 3: System.out.println("num: 3"); break;
			case 4: System.out.println("num: 4"); break;
			case 5: System.out.println("num: 5"); break;
			case 6: System.out.println("num: 6"); break;
		}
	}
}
