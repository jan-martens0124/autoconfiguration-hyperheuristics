package examples.ccf;

public class HeuristicConfiguration {

	private double iom, dos;

	public HeuristicConfiguration(double iom, double dos) {
		this.iom = iom;
		this.dos = dos;
	}

	public double getIom() {
		return iom;
	}

	public double getDos() {
		return dos;
	}

	public void setIom(double iom) {
		this.iom = iom;
	}

	public void setDos(double dos) {
		this.dos = dos;
	}
}
