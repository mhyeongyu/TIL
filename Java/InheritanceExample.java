package chap07;


class CellPhone {
	
	String model;
	String color;
	
	void powerOn() {System.out.println("PowerON!!"); }
	void powerOff() {System.out.println("PowerOFF!"); }
	void bell() {System.out.println("RingDingDong"); }
	void sendVoice(String message) {System.out.println("me:" + message); }
	void receiveVoice(String message) {System.out.println("you:" + message); }
	void hangUp() {System.out.println("finish"); }

}

class DmbCellPhone extends CellPhone {
	
	int channel;
	
	DmbCellPhone(String model, String color, int channel) {
		this.model = model;
		this.color = color;
		this.channel = channel;
	}
	
	void trunOnDmb() {System.out.println("channel " + channel); }
	void changeChannelDmb(int channel) {System.out.println("Change " + channel + " channel"); }
	void turnOffDmb() {System.out.println("Dmb Off!"); }
	
}


public class InheritanceExample {

	public static void main(String[] args) {
		
		DmbCellPhone dmbCellPhone = new DmbCellPhone("apple", "white", 17);
		
		System.out.println("model: " + dmbCellPhone.model);
		System.out.println("color: " + dmbCellPhone.color);
		System.out.println("channel: " + dmbCellPhone.channel);
		
		dmbCellPhone.powerOn();
		dmbCellPhone.bell();
		dmbCellPhone.sendVoice("Hello??");
		dmbCellPhone.receiveVoice("Hello, apple");
		dmbCellPhone.sendVoice("Hi, samsung");
		dmbCellPhone.hangUp();
		
		dmbCellPhone.trunOnDmb();
		dmbCellPhone.changeChannelDmb(27);
		dmbCellPhone.turnOffDmb();

	}

}
