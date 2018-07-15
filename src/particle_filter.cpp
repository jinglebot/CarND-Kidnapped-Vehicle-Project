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
	default_random_engine gen;
	is_initialized = true;
	num_particles = 100;
	particles.resize(0);
	
	// Set standard deviations for x, y, and theta
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
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
		p.weight = 1./num_particles;
		particles.push_back(p);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	is_initialized = true;

	// Set standard deviations for x, y, and theta
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	for (auto &p : particles) {

		if (fabs(yaw_rate) < 0.00001) {  
    		p.x += velocity * delta_t * cos(p.theta);
    		p.y += velocity * delta_t * sin(p.theta);
    	} else {
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
		}

		// This line creates a normal (Gaussian) distribution for x, y and theta
		normal_distribution<double> dist_x(p.x, std_x);
		normal_distribution<double> dist_y(p.y, std_y);
		normal_distribution<double> dist_theta(p.theta, std_theta);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
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

	// initialize weights vector 
	weights.resize(0);

	// calculate normalization term
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));
	
	for (auto &p : particles) {

		// for each particle's associations of observations to landmarks
		vector <double> x_assoc;
		vector <double> y_assoc;
		vector <int> id_assoc;
		double final_weight = 1.0;

		// transform observations to map coordinates
		vector<LandmarkObs> transformed_observations;
		for (auto &c : observations){
			LandmarkObs m;
			m.x = p.x + cos(p.theta) * c.x - sin(p.theta) * c.y;
			m.y = p.y + sin(p.theta) * c.x + cos(p.theta) * c.y;
			m.id = c.id;
			transformed_observations.push_back(m);
		}

		// choose only landmarks within sensor range
		vector <LandmarkObs> predicted;
		for (auto &lm : map_landmarks.landmark_list) {
			double distance_lm = dist(lm.x_f, lm.y_f, p.x, p.y);
			if (distance_lm < sensor_range) {
				LandmarkObs sr;
				sr.x = lm.x_f;
				sr.y = lm.y_f;
				sr.id = lm.id_i;
				predicted.push_back(sr);
			}
		}

		// calculate nearest neighbor
		for (auto &ob : transformed_observations) {
			double mu_x = predicted[0].x;
			double mu_y = predicted[0].y;
			int mu_id = predicted[0].id;
			for (auto &lm : predicted) {
				double distance_lm = dist(lm.x, lm.y, ob.x, ob.y);
				double distance_mu = dist(mu_x, mu_y, ob.x, ob.y);
				if (distance_lm < distance_mu) {
					mu_x = lm.x;
					mu_y = lm.y;
					mu_id = lm.id;
				}
			}

			// associate the nearest landmark to the particle
			x_assoc.push_back(mu_x);
			y_assoc.push_back(mu_y);
			id_assoc.push_back(mu_id);

			// calculate nearest landmark weight
			// calculate exponent
			double exponent = pow((ob.x - mu_x), 2)  / (2 * sig_x * sig_x) + pow((ob.y - mu_y), 2) / (2 * sig_y * sig_y);

			// calculate weight using normalization terms and exponent
			double weight = gauss_norm * exp(-exponent);

			// calculate final weight for this particle
			final_weight *= weight;
		}
		p.weight = final_weight;
		p = SetAssociations(p, id_assoc, x_assoc, y_assoc);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// normalize weights vector 
	double sum_of_wts = 0.0;
	for (auto &p : particles) {
		sum_of_wts += p.weight;
	}
	for (auto &p : particles) {
		p.weight /= sum_of_wts;
		weights.push_back(p.weight);
	}
	// cout << weights.size() << endl;

    // get maximum of weights vector
    vector<double>::iterator it;
    it = max_element(weights.begin(), weights.end());
    double max_w = *it;
    
	vector <Particle> new_particles;
	// do until new_particles length == num_particles
	int index = rand() % num_particles;
	for (int i = 0; i < num_particles; ++i) {
		double beta = ((double) rand() / (RAND_MAX)) * 2.0 * max_w;
		while (weights[index] < beta) {
			beta -= weights[index];
			index += 1;
			if (index >= num_particles) index = 0;
		}
		new_particles.push_back(particles[index]);
	}

	particles.resize(num_particles);
	particles = new_particles;
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

