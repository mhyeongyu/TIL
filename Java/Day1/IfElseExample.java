package chap04;

public class IfElseExample {

	public static void main(String[] args) {
		int score = (int) (Math.random() * 30) + 70;
		
		if(score>=90) {
			System.out.println("Score:"+ score);
			System.out.println("Class: A");
		} else if(score>=80) {
			System.out.println("Socre:"+ score);
			System.out.println("Class: B");
		} else {
			System.out.println("Socre:"+ score);
			System.out.println("Class: C");
		}
	}
}
