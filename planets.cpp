#include <arrayfire.h>
#include <stdio.h>
#include <math.h>

using namespace af;

static size_t num_planets = 4;
static array positions[2];
static array velocities[2];
static array acceleration;
static array masses;
static double timestep = 5000.0; //s
static double t2; //timestep**2
static const double g = 6.674e-11; //Nm^2/kg^2

static void step(int);
static void calculate_acceleration(int);

int main(int argc, char **argv){
	//initialize
	t2 = timestep*timestep;

	//configure simulation time in seconds
	double time = 3e6;
	int steps = (int)(time/timestep);

	dim_t num_elements = 3*num_planets;

	//configure intial positions, velocities and masses
	const double init_positions[num_elements]={0,0,0, 2.944e7,-3.981e8, 3.081e7, 0,0,1.0e8, 0.5e8,0.5e8,0.5e8};
	const double init_velocities[num_elements]={0,0,0, 981.2,98.93,-45.13, 0,0,0, 0,0,0};
	const double init_masses[num_planets] = {6e24, 7.4e22, 0.5, 0.1};
	masses = array(num_planets, init_masses);

	//two-step position/velocity storage. Even steps go into position[0], odd steps go into position[1]
	positions[0] = array(num_planets*3, init_positions);
	positions[1] = array(num_planets*3, init_positions);
	velocities[0] = array(num_planets*3, init_velocities)*timestep;
	velocities[1] = array(num_planets*3, init_velocities)*timestep;
	
	//initialize acceleration matrix
	acceleration = constant(0.0, num_planets*3, num_planets*3, f64);

	//do simulation
	double *buffer;
	for(int i=0;i<steps;++i){
		step(i);
		buffer = positions[i%2].host<double>();
		for(int k = 0; k < 3*num_planets; ++k){
			printf("%.5f\t",buffer[k]);
		}
		printf("\n");
	}
	return 0;
}

static void step(int time_index){
	//calculate the acceleration matrix
	calculate_acceleration(time_index);
	//update positions
	//note: velocities are already stored times timestep to avoid unneccessary multiplications
	positions[time_index%2] = matmul(0.5*acceleration*t2+identity(num_planets*3,num_planets*3),positions[1-time_index%2]) + velocities[1-time_index%2];
	//update velocities
	velocities[time_index%2] = matmul(acceleration*t2,positions[1-time_index%2]) + velocities[1-time_index%2];
}

//this is where the magic happens
static void calculate_acceleration(int time_index){
	array pos_diff(num_planets*3,num_planets-1,f64);
	for(int i=1;i<num_planets;++i){ //calculate distances in parallel. Entries in pos_diff are a_ij = r_i - r_((i-j)%length), basically
		pos_diff.col(i-1)=positions[1-time_index%2]-shift(positions[1-time_index%2],3*i);
	}
	//calculate |r_i - r_j|^3
	array diff_pow3 = moddims(pow(sqrt(sum(pow(moddims(pos_diff, 3, num_planets, num_planets-1),2),0)),3),num_planets,num_planets-1);
	array acc = constant(0.0, num_planets, num_planets, f64);
	//calculate actual entries for the acceleration matrix
	for(int i=0;i<num_planets;++i){
		for(int j=0;j<num_planets-1;++j){
			acc(i,(i-j-1+num_planets)%num_planets) = masses((i-j-1+num_planets)%num_planets)/diff_pow3(i,j);
		}
		acc(i,i) = -sum(acc.row(i));
	}
	acc *= g;
	//inflate acceleration matrix because we store all three coordinates in one array. It might actually be faster to have three separate arrays (for the cost of a slightly more complicated distance calculation)
	acceleration = moddims(reorder(tile(acc,1,1,3,3),2,0,3,1),num_planets*3,num_planets*3)*tile(identity(3,3),num_planets,num_planets);
}
