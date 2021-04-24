package chap04;

public class ForExamlpel {

	public static void main(String[] args) {
		int sum = 0;
		int i = 0;
		
		for (i=1; i<=100; i++) {
			sum += i;
		}
		System.out.println("1 ~ "+ (i-1) + "합: " + sum);
		
		//구구단 
		for (int m=2; m<=9; m++) {
			System.out.println("#### "+m+"단 ####");
			for (int n=1; n<=9; n++) {
				System.out.println(m+"x"+n+" = "+m*n);
			}
		}
	}
}
