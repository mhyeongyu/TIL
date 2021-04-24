package chap06;

class CarMethod{
	//field
	int gas;
	
	//constructor 생략
	
	//method1
	void setGas(int gas) {
		this.gas = gas;
	}
	
	//method2
	boolean isLeftGas() {
		if (gas == 0) {
			System.out.println("gas가 없습니다");
			return false;
		}
		System.out.println("gas가 있습니다");
		return true;
	}
	
	//method3
	void run() {
		while(true) {
			if(gas > 0) {
				System.out.println("gas 잔량: "+ gas);
				gas -= 1;
			} else {
				System.out.println("멈춥니다");
				return;
			}
		}
	}
}

public class MethodExample {
	public static void main(String[] args) {
		CarMethod myCar = new CarMethod();
		
		myCar.setGas(7);
		
		boolean gasState = myCar.isLeftGas();
		
		if (gasState) {
			System.out.println("start");
			myCar.run();
		}
		
		if (myCar.isLeftGas()) {
			System.out.println("continue");
		} else {
			System.out.println("finish");
		}
	}

}