package examples.ccf;

/** 
 * This class represents the configuration settings for a heuristic.
 * 
 * @param iom: intensity of mutation
 * @param dos: diversity of search
 * 
 * @author Warren G Jackson (warren.jackson1@nottingham.ac.uk)
 */

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
