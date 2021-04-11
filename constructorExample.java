package chap06;


class People{
	String nation = "korea";
	String name;
	String number;
	
	People(String n, String nn){
		name = n;
		number = nn;
	}
}

public class constructorExample {

	public static void main(String[] args) {
		People p1 = new People("Kim", "010-1234-5678");
		System.out.println(p1.nation);
		System.out.println(p1.name);
		System.out.println(p1.number);
		
		System.out.println();
		
		People p2 = new People("Lee", "010-5678-1234");
		System.out.println(p2.nation);
		System.out.println(p2.name);
		System.out.println(p2.number);
		
	}

}
