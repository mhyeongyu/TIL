package chap06;


class Car {
	String company = "tesla";
	String color = "white";
	int maxSpeed = 200;
	int speed = 0;
}

public class fieldExample {

	public static void main(String[] args) {
		Car myCar = new Car();
		
		System.out.println(myCar.company);
		System.out.println(myCar.color);
		System.out.println(myCar.maxSpeed);
		System.out.println(myCar.speed);
		
		myCar.speed = 100;
		System.out.println(myCar.speed);

	}

}
