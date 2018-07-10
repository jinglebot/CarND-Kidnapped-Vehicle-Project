/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 20;
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i + 1;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.;
		//p.associations;
		//p.sense_x;
		//p.sense_y;
		particles.push_back(p);
		//cout << "p.id: " << p.id <<  "\tp.x: " << p.x << "\tp.y: " << p.y << "\tp.theta: " << p.theta << endl;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	for (auto &p : particles) {
		if (yaw_rate > 0.001) {
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;

			// This line creates a normal (Gaussian) distribution for x, y and theta
			normal_distribution<double> dist_x(p.x, std_x);
			normal_distribution<double> dist_y(p.y, std_y);
			normal_distribution<double> dist_theta(p.theta, std_theta);

			p.x = dist_x(gen);
			p.y = dist_y(gen);
			p.theta = dist_theta(gen);
			// cout << "p.id: " << p.id <<  "\tp.x: " << p.x << "\tp.y: " << p.y << "\tp.theta: " << p.theta << endl;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// resize weights vector = number of particles
	weights.resize(num_particles, 0);

	// calculate normalization term
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));
	
	for (auto &p : particles) {

		// ***********************************
		// FIND TRANSFORMED MAP COORDINATES
		// ***********************************

		// for each particle's associations of observations to landmarks
		vector <double> x_assoc;
		vector <double> y_assoc;
		vector <int> id_assoc;
		double final_weight = 0;

		for (auto &c : observations){
			LandmarkObs m;
			m.x = p.x + cos(p.theta) * c.x - sin(p.theta) * c.y;
			m.y = p.y + sin(p.theta) * c.x + cos(p.theta) * c.y;
			m.id = c.id;
		
			// **************************
			// FIND THE NEAREST LANDMARK
			// **************************
			// find the landmark associated with this observation
			double mu_x = map_landmarks.landmark_list[0].x_f;
			double mu_y = map_landmarks.landmark_list[0].y_f;
			int mu_id = map_landmarks.landmark_list[0].id_i;
			for (auto &lm : map_landmarks.landmark_list) {
				// choose only landmarks within sensor range
				double distance_x = fabs(lm.x_f - m.x);
				double distance_y = fabs(lm.y_f - m.y);
				if ((distance_x < sensor_range) && (distance_y < sensor_range)) {
					// the nearest neighbor can be closest x-wise or y-wise, get the min distance sum
					if ((distance_x + distance_y) < (fabs(mu_x - m.x) + fabs(mu_y - m.y))) {
						mu_x = lm.x_f;
						mu_y = lm.y_f;
						mu_id = lm.id_i;
					}		
				}
			}

			// ***********************************************
			// ASSOCIATE THE NEAREST LANDMARK TO THE PARTICLE
			// ***********************************************
			x_assoc.push_back(mu_x);
			y_assoc.push_back(mu_y);
			id_assoc.push_back(mu_id);

			// **************************************
			// CALCULATE THE NEAREST LANDMARK WEIGHT
			// **************************************

			// calculate exponent
			double exponent = pow((m.x - mu_x), 2)  / (2 * sig_x * sig_x) + pow((m.y - mu_y), 2) / (2 * sig_y * sig_y);

			// calculate weight using normalization terms and exponent
			double weight = gauss_norm * exp(-exponent);

			// calculate final weight for this particle
			final_weight *= weight;
		}
		p.weight = final_weight;
		weights.push_back(p.weight);
		p = SetAssociations(p, id_assoc, x_assoc, y_assoc);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
