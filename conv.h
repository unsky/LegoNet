#pragma once

#include "layer.h"
//卷积层的实现
//第一部分：卷积核的初始化这里使用随机初始化。
//卷积的前向
//卷积的后向
//卷积的梯度计算
#pragma pack(push, 1)
struct conv_layer_t
{
	layer_type type = layer_type::conv;//层类型
	tensor_t<float> grads_in; //梯度（单个核）
	tensor_t<float> in;//输入map
	tensor_t<float> out;//输出map
	std::vector<tensor_t<float>> filters;//卷积核
	std::vector<tensor_t<gradient_t>> filter_grads;//所有的梯度
	uint16_t stride;
	uint16_t extend_filter;
	//初始化参数 stride 步长， extend_filter,卷积核的大小，number_filters 输出大小 in_size输入大小
	conv_layer_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out(
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ )
		{
			tensor_t<float> t( extend_filter, extend_filter, in_size.z );

			int maxval = extend_filter * extend_filter * in_size.z;

			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in_size.z; z++ )
						t( i, j, z ) = 1.0f / maxval * rand() / float( RAND_MAX );//使用rand()产生随机高斯
			filters.push_back( t );
		}
		for ( int i = 0; i < number_filters; i++ )
		{
			tensor_t<gradient_t> t( extend_filter, extend_filter, in_size.z );
			filter_grads.push_back( t );
		}

	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size() - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()//计算前向卷积
	{
		for ( int filter = 0; filter < filters.size(); filter++ )
		{
			tensor_t<float>& filter_data = filters[filter];
			for ( int x = 0; x < out.size.x; x++ )
			{
				for ( int y = 0; y < out.size.y; y++ )
				{
					point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float sum = 0;
					for ( int i = 0; i < extend_filter; i++ )
						for ( int j = 0; j < extend_filter; j++ )
							for ( int z = 0; z < in.size.z; z++ )
							{
								float f = filter_data( i, j, z );
								float v = in( mapped.x + i, mapped.y + j, z );
								sum += f*v;
							}
					out( x, y, filter ) = sum;
				}
			}
		}
	}

	void fix_weights()//计算后向传播
	{
		for ( int a = 0; a < filters.size(); a++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						float& w = filters[a].get( i, j, z );
						gradient_t& grad = filter_grads[a].get( i, j, z );
						w = update_weight( w, grad );//更新权重
						update_gradient( grad );//更新梯度
					}
	}

	void calc_grads( tensor_t<float>& grad_next_layer )//计算后向梯度
	{

		for ( int k = 0; k < filter_grads.size(); k++ )
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads[k].get( i, j, z ).grad = 0;
		}

		for ( int x = 0; x < in.size.x; x++ )
		{
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;
							for ( int k = rn.min_z; k <= rn.max_z; k++ )
							{
								int w_applied = filters[k].get( x - minx, y - miny, z );
								sum_error += w_applied * grad_next_layer( i, j, k );
								filter_grads[k].get( x - minx, y - miny, z ).grad += in( x, y, z ) * grad_next_layer( i, j, k );
							}
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
		}
	}
};
#pragma pack(pop)