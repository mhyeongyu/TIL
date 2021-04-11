package chap05;

public class ArrayExample {

	public static void main(String[] args) {
		int[] scores = {83, 90, 87};
		
		System.out.println(scores[0]);
		System.out.println(scores[1]);
		
		int sum = 0;
		for (int i=0; i<3; i++) {
			sum += scores[i];
			System.out.println(sum);
		}
		double avg = (double) sum/3;
		System.out.println(avg);
		
		int [] scores1;
		scores1 = new int[] {83, 90, 87};
		int sum1 = 0;
		for (int i=0; i<3; i++) {
			sum1 += scores1[i];
			System.out.println(sum1);
		}
	
	
	}
	
}
