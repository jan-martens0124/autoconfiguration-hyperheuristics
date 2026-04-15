package examples.ccf;

/** 
 * This class represents the data associated with a heuristic, including 
 * @param timeLastApplied: the time it was last applied
 * @param: previousApplicationDuration: the duration of its previous application
 * @param f_delta: the objective function value change after calling the heuristic
 * 
 * @author Warren G Jackson (warren.jackson1@nottingham.ac.uk)
 */


public class HeuristicData {

	private long timeLastApplied;
	
	private long previousApplicationDuration;
	
	private double f_delta;
	
	public HeuristicData(long currentTime) {
		
		this.timeLastApplied = currentTime;
		this.f_delta = -Double.MAX_VALUE;
		this.previousApplicationDuration = 0;
	}

	public long getTimeLastApplied() {
		return timeLastApplied;
	}

	public void setTimeLastApplied(long timeLastApplied) {
		this.timeLastApplied = timeLastApplied;
	}

	public double getF_delta() {
		return f_delta;
	}

	public void setF_delta(double f_delta) {
		this.f_delta = f_delta;
	}

	public long getPreviousApplicationDuration() {
		return previousApplicationDuration;
	}

	public void setPreviousApplicationDuration(long previousApplicationDuration) {
		this.previousApplicationDuration = previousApplicationDuration;
	}
}
