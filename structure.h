#pragma once
/* 该段程序负责前向传播和后向传播的层数判断以及梯度计算
*/
#include "tensor_t.h"
#include "optimization_method.h"
#include "fc.h"
#include "max_pool.h"
#include "relu.h"
#include "conv.h"
#include "dropout.h"
static void calc_grads( layer_t* layer, tensor_t<float>& grad_next_layer )//前向
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::relu:
			((relu_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::dropout_layer:
			((dropout_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		default:
			assert( false );
	}
}

static void fix_weights( layer_t* layer )//后向
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->fix_weights();
			return;
		case layer_type::relu:
			((relu_layer_t*)layer)->fix_weights();
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->fix_weights();
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->fix_weights();
			return;
		case layer_type::dropout_layer:
			((dropout_layer_t*)layer)->fix_weights();
			return;
		default:
			assert( false );
	}
}

static void activate( layer_t* layer, tensor_t<float>& in )//前向
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->activate( in );
			return;
		case layer_type::relu:
			((relu_layer_t*)layer)->activate( in );
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->activate( in );
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->activate( in );
			return;
		case layer_type::dropout_layer:
			((dropout_layer_t*)layer)->activate( in );
			return;
		default:
			assert( false );
	}
}