<%
cfg['libraries'] = ['ceres']
cfg['include_dirs'] = [
	'/usr/local/include/eigen3'
	]
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++1z', '-g', '-Ofast', '-UNDEBUG', '-Wno-misleading-indentation']
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <cmath>
#include <experimental/any>

namespace py = pybind11;

using Eigen::Matrix;
using Eigen::RowMajor;

template<typename T> using Point = Matrix<T, 1, 3, RowMajor>;
template<typename T> using Points = Matrix<T, Eigen::Dynamic, 3, RowMajor>;
template<typename T> using Pixel = Matrix<T, 1, 2, RowMajor>;
template<typename T> using Pixels = Matrix<T, Eigen::Dynamic, 2, RowMajor>;
template<typename T> using Tvec = Matrix<T, 1, 3, RowMajor>;
template<typename T> using Rvec = Matrix<T, 1, 3, RowMajor>;
template<typename T> using Quat = Matrix<T, 1, 4, RowMajor>;
template<typename T> using Projection = Matrix<T, 3, 3, RowMajor>;


template <typename T>
Point<T> transform_point(const Point<T>& point, const Rvec<T>& rvec, const Tvec<T>& tvec) {
	Point<T> p;
	ceres::AngleAxisRotatePoint(rvec.data(), point.data(), p.data());
	p += tvec;
	return p;
}

template <typename T>
Point<T> transform_point(const Point<T>& point, const Quat<T>& quat, const Tvec<T>& tvec) {
	Point<T> p;
	ceres::QuaternionRotatePoint(quat.data(), point.data(), p.data());
	p += tvec;
	return p;
}

template <typename T, typename R>
Pixel<T> project_point(const Point<T>& point, const R& rvec,
		const Tvec<T>& tvec, const Projection<T>& camera) {
	auto p = transform_point(point, rvec, tvec);
	return (camera*p.transpose()).colwise().hnormalized();
}

template <typename T, typename R>
Pixel<T> project_point(const Point<T>& point, const R& rvec,
		const Tvec<T>& tvec, const T* cam_p) {
	/*auto camera = Projection<T>::Zero();
	camera(0, 0) = cam_p[0];
	camera(0, 2) = cam_p[1];
	camera(1, 1) = cam_p[2];
	camera(1, 2) = cam_p[3];
	camera(2, 2) = T(1.0);*/
	auto p = transform_point(point, rvec, tvec);
	Pixel<T> ret;
	ret(0,0) = p(0,0)/p(0,2)*cam_p[0] + cam_p[1];
	ret(0,1) = p(0,1)/p(0,2)*cam_p[2] + cam_p[3];
	return ret;
	//return (camera*p.transpose()).colwise().hnormalized();
}


template <typename T>
T sign_func(const T& x)
{
	return x < T(0.0) ? T(-1.0) : T(+1.0);
}

struct PointReprojectionError {
	Pixels<double> observed;
	Projection<double> camera;

	PointReprojectionError(const Pixels<double>& observed, Projection<double>& camera)
		:observed(observed), camera(camera) {}
	
	template<typename T>
	bool operator()(
			const T* const cr,
			const T* const ct,
			const T* const point_,
			T* residuals) const {
		

		auto projected = project_point(
				Point<T>(point_),
				Quat<T>(cr),
				Tvec<T>(ct),
				camera.cast<T>().eval());
		auto error = (projected - observed.cast<T>().eval()).eval();
		Eigen::Map<decltype(error)> resid(residuals);
		resid = error;
		return true;
	}
	
	static auto Create(auto&&... params) {
		using Me = PointReprojectionError;
		return new ceres::AutoDiffCostFunction<Me, 2, 4, 3, 3>(new Me(params...));
	}

};

struct MarkerReprojectionError {
	const Pixels<double>& observed;
	const Points<double>& geometry;
	const Projection<double>& camera;

	MarkerReprojectionError(const Pixels<double>& observed,
			const Points<double>& geometry, Projection<double>& camera)
		:observed(observed), geometry(geometry), camera(camera) {}
	
	template<typename T>
	bool operator()(
			const T* const cam,
			const T* const cr,
			const T* const ct,
			const T* const pr,
			const T* const pt,
			const T* const ps,
			T* residuals) const {
		auto const n = geometry.rows();
		Quat<T> pRot(pr);
		Tvec<T> pTrans(pt);
		Quat<T> cRot(cr);
		Tvec<T> cTrans(ct);
		//auto cm = camera.cast<T>().eval();

		for(int i=0; i < n; ++i) {
			auto p = (geometry.row(i).cast<T>()*exp(T(*ps))).eval();
			auto point = transform_point(p, pRot, pTrans);
			auto depth = point(0, 2);
			//if(depth <= T(0.0)) return false;
			auto projected = project_point(point, cRot, cTrans, cam);
			auto error = (projected - observed.row(i).cast<T>().eval()).eval();
			auto p_i = i*(error.cols());
			//residuals[p_i] = depth > T(0.0) ? T(0.0) : depth*T(100);
			Eigen::Map<decltype(error)> resid(residuals + p_i);
			resid = error;
		}
		return true;
	}
	
	static auto Create(const auto& observed, auto&&... params) {
		using Me = MarkerReprojectionError;
		return new ceres::AutoDiffCostFunction<Me, ceres::DYNAMIC, 4, 4, 3, 4, 3, 1>(
				new Me(observed, params...), observed.rows()*(observed.cols()));
	}

};

struct CheiralityConstraint {
	template<typename T>
	bool operator()(
			const T* const cr,
			const T* const ct,
			const T* const point_,
			T* residuals) const {
		auto depth = transform_point(Point<T>(point_), Quat<T>(cr), Tvec<T>(ct))(0, 2);
		residuals[0] = depth > T(0.0) ? T(0.0) : depth*T(10);
		return true;
	}
	
	static auto Create(auto&&... params) {
		using Me = CheiralityConstraint;
		return new ceres::AutoDiffCostFunction<Me, 1, 4, 3, 3>(new Me(params...));
	}

};


using Frame = std::map<int, Pixels<double>>;
using Frames = std::map<int, Frame>;
using Pose = std::tuple<Rvec<double>, Tvec<double>>;



template<typename T>
Rvec<T> _angular_velocity(const Quat<T>& q0, Quat<T> q1, T dt) {
	const T* quat0 = q0.data();
	static const T neg(-1.0);
	if(q0.dot(q1) < T(0.0)) q1 *= neg;
	q1.block(0, 1, 1, 3) *= neg;
	T* quat1 = q1.data();
	//quat1[1] *= neg; quat1[2] *= neg; quat1[3] *= neg;
	T diff[4];
	ceres::QuaternionProduct(quat0, quat1, diff);
	Rvec<T> result;
	ceres::QuaternionToAngleAxis(diff, result.data());
	result /= dt;
	return result;
}

struct RelativePoseError {
	double ts0, ts1, ts2;

	RelativePoseError(double ts0, double ts1, double ts2): ts0(ts0), ts1(ts1), ts2(ts2) {}
	
	template<typename T>
	bool operator()(
			const T* const r0, const T* const t0,
			const T* const r1, const T* const t1,
			const T* const r2, const T* const t2,
			T* residuals) const {
		Quat<T> rvec0(r0); Tvec<T> tvec0(t0);
		Quat<T> rvec1(r1); Tvec<T> tvec1(t1);
		Quat<T> rvec2(r2); Tvec<T> tvec2(t2);
		auto dt1 = ts1 - ts0;
		auto dt2 = ts2 - ts1;
		
		//auto rspeed1 = _angular_velocity(rvec0, rvec1, T(dt1));
		//auto rspeed2 = _angular_velocity(rvec1, rvec2, T(dt2));
		auto rspeed1 = ((rvec1 - rvec0)/T(dt1)).eval();
		auto rspeed2 = ((rvec2 - rvec1)/T(dt2)).eval();
		auto raccel = ((rspeed2 - rspeed1)/T(dt2)).eval();
		
		auto tspeed1 = ((tvec1 - tvec0)/T(dt1)).eval();
		auto tspeed2 = ((tvec2 - tvec1)/T(dt2)).eval();
		auto taccel = ((tspeed2 - tspeed1)/T(dt2)).eval();
		
		Eigen::Map<decltype(raccel)> rresid(residuals); rresid = rspeed1/T(1.0);
		Eigen::Map<decltype(taccel)> tresid(residuals+4); tresid = taccel/T(10.0);
		
		return true;
	}
	
	static auto Create(auto&&... params) {
		using Me = RelativePoseError;
		//return new ceres::AutoDiffCostFunction<Me, 3+3, 4,3, 4,3, 4,3>(new Me(params...));
		return new ceres::AutoDiffCostFunction<Me, 4+3, 4,3, 4,3, 4,3>(new Me(params...));
	}

};

struct PoseDifference {
	Tvec<double> _tvec0;
	Quat<double> _rvec0;
	PoseDifference(const Quat<double>& rvec0, const Tvec<double>& tvec0): _tvec0(tvec0), _rvec0(rvec0)
	{
		assert(tvec0.data() != _tvec0.data());
		assert(rvec0.data() != _rvec0.data());
	}
	
	template<typename T>
	bool operator()(
			const T* const r, const T* const t,
			T* residuals) const {
		Quat<T> rvec(r); Tvec<T> tvec(t);
		auto rvec0 = _rvec0.cast<T>().eval();
		auto tvec0 = _tvec0.cast<T>().eval();
		
		auto rdiff = (_angular_velocity(rvec0, rvec, T(1.0))*T(180.0/M_PI*10.0)).eval();
		auto tdiff = ((tvec - tvec0)*T(10000.0)).eval();
		
		Eigen::Map<decltype(rdiff)> rresid(residuals); rresid = rdiff;
		Eigen::Map<decltype(tdiff)> tresid(residuals+3); tresid = tdiff;
		return true;
	}
	
	static auto Create(auto&&... params) {
		using Me = PoseDifference;
		return new ceres::AutoDiffCostFunction<Me, 3+3, 4,3>(new Me(params...));
	}

};

auto smooth_poses(std::map<int, Pose> camera_poses, std::map<int, double> frame_times) {
	ceres::Problem problem;
	std::vector<int> pose_idx;
	std::transform(camera_poses.begin(), camera_poses.end(), std::back_inserter(pose_idx),
			[](const auto& item) {return item.first;});
	bool robust_camera = false;
	std::map<int, Quat<double>> quats;
	for(auto const& pose : camera_poses) {
		ceres::AngleAxisToQuaternion(std::get<0>(pose.second).data(), quats[pose.first].data());
	}
	
	int n = pose_idx.size();
	for(auto i=0; i < n; ++i) {

		auto prev = pose_idx[i > 0 ? i - 1 : 2];
		auto current = pose_idx[i];
		auto next = pose_idx[i < n - 1 ? i + 1 : i - 2];

		auto cost = RelativePoseError::Create(
				frame_times[prev],
				frame_times[current],
				frame_times[next]);
		
		auto pose = &camera_poses.at(prev);
		double *t_prev = std::get<1>(*pose).data();
		double *r_prev = quats[prev].data();
		
		pose = &camera_poses.at(current);
		double *t_current = std::get<1>(*pose).data();
		double *r_current = quats[current].data();
		problem.AddResidualBlock(
				PoseDifference::Create(quats[current], std::get<1>(*pose)),
				new ceres::HuberLoss(1.0), r_current, t_current);

		pose = &camera_poses.at(next);
		double *t_next = std::get<1>(*pose).data();
		double *r_next = quats[next].data();
		auto loss = robust_camera?(new ceres::HuberLoss(1.0)):NULL;
		problem.AddResidualBlock(cost, loss,
				r_prev, t_prev,
				r_current, t_current,
				r_next, t_next);

	}

	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	//} else {
	//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//}
	options.max_num_iterations = 1000;
	options.use_nonmonotonic_steps = true;
	//options.max_solver_time_in_seconds = 10.0;
	//options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::map<int, double> camera_costs;
	ceres::Problem::EvaluateOptions evalopts;
	evalopts.apply_loss_function = true;
	for(auto& pose : camera_poses) {
		ceres::QuaternionToAngleAxis(
				quats[pose.first].data(),
				std::get<0>(pose.second).data());
		//evalopts.parameter_blocks = {quats[pose.first].data(), std::get<1>(pose.second).data()};
		//evalopts.residual_blocks = {poseResiduals.at(pose.first)};
		//problem.Evaluate(evalopts, &camera_costs[pose.first], NULL, NULL, NULL);
	}
	std::cout << summary.FullReport() << "\n";
	return std::make_tuple(camera_poses, camera_costs);


}

using MarkerPose = std::tuple<Rvec<double>, Tvec<double>, double>;

auto marker_ba(
		Projection<double> camera,
		Frames frames,
		std::map<int, double> frame_times,
		std::map<int, Pose> camera_poses,
		std::map<int, MarkerPose> marker_poses,
		std::map<int, Points<double>> marker_geometry,
		std::set<int> fixed_features,
		bool fix_cameras=false,
		bool fix_features=false,
		bool smooth_camera=false,
		bool robust_camera=true
		) {
	ceres::Problem problem;
	std::map<int, Quat<double>> quats;
	std::map<int, Quat<double>> mquats;
	double cam_p[4];
	cam_p[0] = camera(0, 0);
	cam_p[1] = camera(0, 2);
	cam_p[2] = camera(1, 1);
	cam_p[3] = camera(1, 2);
	problem.AddParameterBlock(cam_p, 4,
		new ceres::SubsetParameterization(4, {1, 3})
		);
	problem.SetParameterBlockConstant(cam_p);

	for(auto& pose : marker_poses) {
		auto tmp = mquats[pose.first].data();
		ceres::AngleAxisToQuaternion(
				std::get<0>(pose.second).data(),
				tmp);
		problem.AddParameterBlock(tmp, 4, new ceres::QuaternionParameterization);
		auto scale = &std::get<2>(pose.second);
		problem.AddParameterBlock(scale, 1);
		//problem.SetParameterLowerBound(scale, 0, log(0.5));
		//problem.SetParameterUpperBound(scale, 0, log(2.0));
		problem.SetParameterBlockConstant(scale);
	}

	for(auto& frame : frames) {
		if(!camera_poses.count(frame.first)) continue;
		auto* pose = &camera_poses.at(frame.first);
		double *_rvec = std::get<0>(*pose).data();
		double *rvec = quats[frame.first].data();
		ceres::AngleAxisToQuaternion(_rvec, rvec);
		problem.AddParameterBlock(rvec, 4, new ceres::QuaternionParameterization);
		double *tvec = std::get<1>(*pose).data();
		problem.AddParameterBlock(rvec, 4);
		problem.AddParameterBlock(tvec, 3);
		if(fix_cameras) {
			problem.SetParameterBlockConstant(rvec);
			problem.SetParameterBlockConstant(tvec);
		}

		for(auto& marker : frame.second) {
			if(!marker_geometry.count(marker.first)) continue; 
			if(!marker_poses.count(marker.first)) continue; 
			auto& mpoints = marker_geometry.at(marker.first);
			auto& mpose = marker_poses.at(marker.first);
			auto cost = MarkerReprojectionError::Create(marker.second, mpoints, camera);
			auto mrvec = mquats[marker.first].data();
			auto mtvec = std::get<1>(mpose).data();
			auto mscale = &std::get<2>(mpose);
			problem.AddResidualBlock(cost,
				new ceres::HuberLoss(1.0),
				cam_p, rvec, tvec, mrvec, mtvec, mscale);
			if(fix_features || fixed_features.count(marker.first)) {
				problem.SetParameterBlockConstant(mrvec);
				problem.SetParameterBlockConstant(mtvec);
				problem.SetParameterBlockConstant(mscale);
			}
			
			/*for(auto i=0; i < marker.second.rows(); ++i) {
				auto cost = PointReprojectionError::Create(marker.second.row(i), camera);
				double *points = mpoints.row(i).data();
				problem.AddResidualBlock(cost,
					new ceres::HuberLoss(1.0),
					rvec, tvec, points);
								if(fix_features or marker.first == reference_id) {
					problem.SetParameterBlockConstant(points);
				}
				
				problem.AddResidualBlock(
					CheiralityConstraint::Create(), NULL,
					rvec, tvec, points);
			}*/
		}
	}

	std::vector<int> pose_idx;
	std::transform(camera_poses.begin(), camera_poses.end(), std::back_inserter(pose_idx),
			[](const auto& item) {return item.first;});
	int n = pose_idx.size();
	for(auto i=0; i < n; ++i) {
		//int prev, current, next;
		auto prev = pose_idx[i > 0 ? i - 1 : 2];
		auto current = pose_idx[i];
		auto next = pose_idx[i < n - 1 ? i + 1 : i - 2];

		auto cost = RelativePoseError::Create(
				frame_times[prev],
				frame_times[current],
				frame_times[next]);
		
		auto pose = &camera_poses.at(prev);
		double *t_prev = std::get<1>(*pose).data();
		double *r_prev = quats[prev].data();
		
		pose = &camera_poses.at(current);
		double *t_current = std::get<1>(*pose).data();
		double *r_current = quats[current].data();

		pose = &camera_poses.at(next);
		double *t_next = std::get<1>(*pose).data();
		double *r_next = quats[next].data();
		auto loss = robust_camera?(new ceres::HuberLoss(0.1)):NULL;

		if(!smooth_camera) continue;
		problem.AddResidualBlock(cost, loss,
				r_prev, t_prev,
				r_current, t_current,
				r_next, t_next);

	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	//options.trust_region_strategy_type = ceres::DOGLEG;
	//options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	//} else {
	//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//}
	options.max_num_iterations = 50;
	//options.use_nonmonotonic_steps = true;
	//options.max_solver_time_in_seconds = 10.0;
	//options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::map<int, double> camera_costs;
	ceres::Problem::EvaluateOptions evalopts;
	evalopts.apply_loss_function = true;
	for(auto& pose : camera_poses) {
		ceres::QuaternionToAngleAxis(
				quats[pose.first].data(),
				std::get<0>(pose.second).data());
	}

	for(auto& pose : marker_poses) {
		ceres::QuaternionToAngleAxis(
				mquats[pose.first].data(),
				std::get<0>(pose.second).data());
	}

	camera(0, 0) = cam_p[0];
	camera(0, 2) = cam_p[1];
	camera(1, 1) = cam_p[2];
	camera(1, 2) = cam_p[3];
	//camera(2, 2) = T(1.0);
	//std::cout << summary.FullReport() << "\n";
	return std::make_tuple(camera, camera_poses, marker_poses);
}

auto nonlinear_triangulation(
		Projection<double>& camera,
		std::vector<Pose>& poses,
		const std::vector<Pixel<double>>& pixels,
		const Point<double>& est) {
	ceres::Problem problem;
	std::map<int, Quat<double>> quats;
	Point<double> point(est);
	

	for(auto i = 0; i < poses.size(); ++i) {
		auto cost = PointReprojectionError::Create(pixels[i], camera);
		auto quat = quats[i].data();
		ceres::AngleAxisToQuaternion(std::get<0>(poses[i]).data(), quat);
		auto tvec = std::get<1>(poses[i]).data();
		problem.AddResidualBlock(
				cost, new ceres::HuberLoss(1.0),
				quat, tvec, point.data());
		problem.AddResidualBlock(
			CheiralityConstraint::Create(), NULL,
			quat, tvec, point.data());

		problem.SetParameterBlockConstant(quat);
		problem.SetParameterBlockConstant(tvec);
	}

	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << "Triangulation" << std::endl;
	std::cout << summary.FullReport() << "\n";

	return point;
}

auto marker_point_ba(
		Projection<double> camera,
		Frames frames,
		std::map<int, double> frame_times,
		std::map<int, Pose> camera_poses,
		std::map<int, Points<double>> markers,
		int reference_id,
		bool fix_cameras=false,
		bool fix_features=false,
		bool smooth_camera=false,
		bool robust_camera=true
		) {
	ceres::Problem problem;
	std::map<int, Quat<double>> quats;
	for(auto& frame : frames) {
		if(!camera_poses.count(frame.first)) continue;
		auto* pose = &camera_poses.at(frame.first);
		double *_rvec = std::get<0>(*pose).data();
		double *rvec = quats[frame.first].data();
		ceres::AngleAxisToQuaternion(_rvec, rvec);
		problem.AddParameterBlock(rvec, 4, new ceres::QuaternionParameterization);
		double *tvec = std::get<1>(*pose).data();
		
		for(auto& marker : frame.second) {
			if(!markers.count(marker.first)) continue; 
			auto& mpoints = markers.at(marker.first);
			for(auto i=0; i < marker.second.rows(); ++i) {
				auto cost = PointReprojectionError::Create(marker.second.row(i), camera);
				double *points = mpoints.row(i).data();
				problem.AddResidualBlock(cost,
					new ceres::HuberLoss(1.0),
					rvec, tvec, points);
				if(fix_cameras) {
					problem.SetParameterBlockConstant(rvec);
					problem.SetParameterBlockConstant(tvec);
				}

				if(fix_features or marker.first == reference_id) {
					problem.SetParameterBlockConstant(points);
				}
				
				problem.AddResidualBlock(
					CheiralityConstraint::Create(), NULL,
					rvec, tvec, points);
			}
		}
	}

	std::vector<int> pose_idx;
	std::transform(camera_poses.begin(), camera_poses.end(), std::back_inserter(pose_idx),
			[](const auto& item) {return item.first;});
	int n = pose_idx.size();
	for(auto i=0; i < n; ++i) {
		//int prev, current, next;
		auto prev = pose_idx[i > 0 ? i - 1 : 2];
		auto current = pose_idx[i];
		auto next = pose_idx[i < n - 1 ? i + 1 : i - 2];

		auto cost = RelativePoseError::Create(
				frame_times[prev],
				frame_times[current],
				frame_times[next]);
		
		auto pose = &camera_poses.at(prev);
		double *t_prev = std::get<1>(*pose).data();
		double *r_prev = quats[prev].data();
		
		pose = &camera_poses.at(current);
		double *t_current = std::get<1>(*pose).data();
		double *r_current = quats[current].data();

		pose = &camera_poses.at(next);
		double *t_next = std::get<1>(*pose).data();
		double *r_next = quats[next].data();
		auto loss = robust_camera?(new ceres::HuberLoss(10.0)):NULL;

		if(!smooth_camera) continue;
		problem.AddResidualBlock(cost, loss,
				r_prev, t_prev,
				r_current, t_current,
				r_next, t_next);

	}

	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	//} else {
	//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//}
	options.max_num_iterations = 1000;
	options.use_nonmonotonic_steps = true;
	//options.max_solver_time_in_seconds = 10.0;
	//options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::map<int, double> camera_costs;
	ceres::Problem::EvaluateOptions evalopts;
	evalopts.apply_loss_function = true;
	for(auto& pose : camera_poses) {
		ceres::QuaternionToAngleAxis(
				quats[pose.first].data(),
				std::get<0>(pose.second).data());
		//evalopts.parameter_blocks = {quats[pose.first].data(), std::get<1>(pose.second).data()};
		//evalopts.residual_blocks = {poseResiduals.at(pose.first)};
		//problem.Evaluate(evalopts, &camera_costs[pose.first], NULL, NULL, NULL);
	}
	std::cout << summary.FullReport() << "\n";
	return std::make_tuple(camera_poses, markers, camera_costs);
}

PYBIND11_PLUGIN(marker_ba) {
using namespace pybind11::literals;
pybind11::module m("marker_ba", "Marker bundle adjustment");
/*m.def("planarPnP", &planarPnP);*/

//m.def("transform_point", &transform_point<double, Rvec<double>>);
//m.def("project_point", &project_point<double, Rvec<double>>);
m.def("marker_point_ba", &marker_point_ba,
		"camera"_a, "frames"_a, "frame_times"_a, "camera_poses"_a, "markers"_a, "reference_id"_a,
		"fix_cameras"_a=false, "fix_features"_a=false, "smooth_camera"_a=false, "robust_camera"_a=true);
m.def("marker_ba", &marker_ba,
		"camera"_a, "frames"_a, "frame_times"_a, "camera_poses"_a, "marker_poses"_a, "markers"_a, "fixed_features"_a,
		"fix_cameras"_a=false, "fix_features"_a=false, "smooth_camera"_a=false, "robust_camera"_a=true);
m.def("smooth_poses", &smooth_poses);
m.def("nonlinear_triangulation", &nonlinear_triangulation);
/*py::class_<MarkerSfm>(m, "MarkerSfm")
	.def(py::init<cv::Mat, cv::Mat, std::tuple<int, int>, int>())
	.def("addFrame", &MarkerSfm::addFrame)
	.def("execute", &MarkerSfm::execute)
;*/
return m.ptr();
}

