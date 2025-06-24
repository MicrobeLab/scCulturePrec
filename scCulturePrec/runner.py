import argparse
import torch
from torch import optim

from scCulturePrec.create_siamese_pairs import create_pairs
from scCulturePrec.model import SiameseFTT, SiameseResNet, SiameseHybrid
from scCulturePrec.wrapper import wrap_train_subfiles, wrap_generate_feature_map, wrap_matcher, elastic_net, elastic_net_pred


def main():
    parser = argparse.ArgumentParser(description="Usage of scCulturePrec.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser for create-pair
    parser_one = subparsers.add_parser('create-pair', help='Create paired-samples for training')
    parser_one.add_argument('--npy-fns', type=str, help='A text file with one .npy file name per line')
    parser_one.add_argument('--half-num', type=int, 
        help='Total number of positive/negative sample pairs', default=100000)
    parser_one.add_argument('--out', type=str, help='Output prefix')
    parser_one.add_argument('--block-size', type=int, help='Number of sample pairs in each output block', default=5000)

    # Subparser for elastic-net
    parser_two = subparsers.add_parser('elastic-net', help='Elastic net for distances to reference')
    parser_two.add_argument('--dist-pos', type=str, help='Input distances of positive samples (.txt)', default=None)
    parser_two.add_argument('--dist-neg', type=str, help='Input distances of negative samples (.txt)', default=None)
    parser_two.add_argument('--model-file', type=str, help='Elastic net model file (.pkl)')
    parser_two.add_argument('--dist-new', type=str, help='Input distances of new samples (.txt)', default=None)
    parser_two.add_argument('--pred-out', type=str, help='Predictions (.csv)', default=None)
    parser_two.add_argument('--train', action='store_true', help='Training elastic net')

    # Subparser for dl-model
    parser_three = subparsers.add_parser('dl-model', 
        help='Siamese network: training, encoding feature vectors, matching (default)')
    parser_three.add_argument('--train', action='store_true', help='Training Siamese network')
    parser_three.add_argument('--encode', action='store_true', help='Encoding feature vectors')
    parser_three.add_argument('--batch-size', type=int, help='Batch size', default=512)
    parser_three.add_argument('--fn', type=str, help='Input filename')
    parser_three.add_argument('--embed-size-spectra', type=int, help='Embedding size for spectral model', default=128) 
    parser_three.add_argument('--embed-size-morphol', type=int, help='Embedding size for morphological model', default=128) 
    parser_three.add_argument('--in-type', type=str, help='morphol/spectra/both', 
        default='both', choices=['morphol', 'spectra', 'both'])
    parser_three.add_argument('--num-morphol', type=int, help='Number of morphological features', default=10) 
    parser_three.add_argument('--input-dir', type=str, help='Input directory', default=None)
    parser_three.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser_three.add_argument('--ckpt', type=str, help='Prefix of checkpoint files', default=None) 
    parser_three.add_argument('--margin', type=float, help='Margin in contrastive loss', default=1.0)
    parser_three.add_argument('--weight', type=str, help='/path/to/weights.pth', default=None) 
    parser_three.add_argument('--feat-db', type=str, help='/path/to/ref_feature_map.h5', default=None) 
    parser_three.add_argument('--out', type=str, help='Output prefix', default=None)

    args = parser.parse_args()
    
    # Call the appropriate function based on the command
    sub_commands = ['create-pair', 'elastic-net', 'dl-model']
    assert args.command in sub_commands, f"'{args.command}' should be one of: create-pair, elastic-net, dl-model"
    if args.command == 'create-pair':
    	create_pairs(npy_fns=args.npy_fns, half_num_samples=args.half_num, out_prefix=args.out, 
            rows_per_file=args.block_size)
    elif args.command == 'elastic-net':
    	if args.train:
    		elastic_net(fn_pos=args.dist_pos, fn_neg=args.dist_neg, output_model=args.model_file)
    	else:
    		elastic_net_pred(model_pkl=args.model_file, dist_txt=args.dist_new, output_csv=args.pred_out)
    elif args.command == 'dl-model':
        cuda = torch.cuda.is_available()
        if in_type == 'morphol':
            model = SiameseFTT(d_numerical=args.num_morphol, categories=None, token_bias=True, 
                    n_layers=1, d_token=200, n_heads=8, d_ffn_factor=2, 
                    attention_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.0, 
                    activation='reglu', prenormalization=True, initialization='kaiming', kv_compression=None, 
                    kv_compression_sharing=None, dim_before=200, embedding_size=args.embed_size_morphol)  
        elif in_type == 'spectra':
            model = SiameseResNet(hidden_sizes=[100]*6, num_blocks=[2]*6, 
                    in_channels=64, embedding_size=args.embed_size_spectra, dim_before=7600)  
        elif in_type == 'both':
            model = SiameseHybrid(num_morphol=args.num_morphol, 
                embed_size_morphol=args.embed_size_morphol, embed_size_spectra=args.embed_size_spectra, 
                dim_before_morphol=200, dim_before_spectra=7600)
        else:
            model = None
        if cuda:
            model.cuda()
        if args.train:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
            wrap_train_subfiles(fn=args.fn, batch_size=args.batch_size, model=model, cuda=cuda, optimizer=optimizer, 
                ckpt=args.ckpt, margin=args.margin, input_dir=args.input_dir)
        elif args.encode:
            wrap_generate_feature_map(fn=args.fn, batch_size=args.batch_size, model=model, cuda=cuda, 
                weight=args.weight, out_prefix=args.out, input_dir=args.input_dir)
        else:
            wrap_matcher(fn=args.fn, batch_size=args.batch_size, model=model, cuda=cuda, 
                weight=args.weight, out_prefix=args.out, feat_db=args.feat_db, input_dir=args.input_dir)


if __name__ == '__main__':
    main()
