#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm.h"
#include <ctype.h>
#include "utility.h"
#include <errno.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))  // Memory Allocation Macro

int print_null(const char *s,...) {return 0;}
static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;

static char *line = NULL;
static int max_line_len;

int ch;  // 0 initialize
int flag = 1; // Initialize a flag variable

static char* readline(FILE *input) {
    int len;
    if (fgets(line, max_line_len, input) == NULL)
        {
        EXIT_FAILURE;
        return NULL; 
        }

    while (strrchr(line, '\n') == NULL) {
        max_line_len *= 2;
        char *new_line = (char *)realloc(line, max_line_len);
        if (new_line == NULL) {
            fprintf(stderr, "realloc hatasi: Bellek yetersiz.\n");
            exit(EXIT_FAILURE);
        }
        line = new_line;
    }
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);

	double *prob_estimates=NULL;
	int j;
    if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else if(svm_type==ONE_CLASS)
		{
			// nr_class = 2 for ONE_CLASS
			prob_estimates = Malloc(double , nr_class);
			//prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"label normal outlier\n");
		}
		else
		{
			int *labels = Malloc(int , nr_class);
			//int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);
			prob_estimates = Malloc(double , nr_class);
			//prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}
	
	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	//line = (char *)malloc(max_line_len*sizeof(char));
	fprintf(output, "  True Label                Predicted Label \n"); 
    fprintf(output, "  ==========                =============== \n\n"); 


	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);

		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				//x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;


 	double true_label;

    if (sscanf(line, "%lf", &true_label) != 1) {
        fprintf(stderr, "Gerçek etiket okuma hatasi.\n");
        exit(EXIT_FAILURE);
    }

    predict_label = svm_predict(model, x);
	int flag1 = 1; 

		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC || svm_type==ONE_CLASS))
		{
			predict_label = svm_predict_probability(model,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
		//	fprintf(output,"\n");
		}
		else
		{
			predict_label = svm_predict(model,x);

			//fprintf(output, " %.17g  [%g]\n ", true_label, predict_label);
		}
		if (true_label != predict_label) {
			
				fprintf(output, "     [%2g]                       [%2g]   False Classify :(   \n", true_label, predict_label);
			}
		else {
				fprintf(output, "     [%2g]                       [%2g] \n", true_label, predict_label);
    }

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
		if(predict_probability)
			free(prob_estimates);		
}

int main(int argc, char **argv) {

    FILE *input, *output;
	int i, heart_rate, breath_rate;
	double rcs;

    if (argc != 4) {

        fprintf(stderr, "Kullanim: %s test_dosyasi model_dosyasi sonuc_dosyasi\n", argv[0]);
        EXIT_FAILURE;
    }
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
		}
	}

	if(i>=argc-2)

		exit(EXIT_FAILURE);; 

	input = fopen(argv[i],"r");

	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(EXIT_FAILURE);
	}

	output = fopen(argv[i+2],"w");

	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = Malloc(struct svm_node , max_nr_attr);
	//x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));

	if(predict_probability)
	{
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	
	else
	{
		if(svm_check_probability_model(model)!=0)

			info("Model supports probability estimates, but disabled in prediction.\n");
	}
		 predict(input,output);
  
    while(flag){

		printf("\n\n Enter the Heart Rate, Breath Rate and RCS Value (Example : 70 20 0.5):\n ");

        int scan_result = scanf("%d %d %lf", &heart_rate, &breath_rate, &rcs);

        if (scan_result != 3)
            EXIT_FAILURE;

        x[0].index = 1;
        x[0].value = heart_rate;
        x[1].index = 2;
        x[1].value = breath_rate;
        x[2].index = 3;
        x[2].value = rcs;
        x[3].index = -1;

		double predict_label = 0;
        int nr_class = svm_get_nr_class(model);

        if (predict_probability) {
		double *prob_estimates = Malloc(double, nr_class);
  		//double *prob_estimates = (double *)malloc(nr_class * sizeof(double));

	    if (prob_estimates == NULL) {

				fprintf(stderr, "Bellek ayirma hatasi: prob_estimates\n");
				exit(EXIT_FAILURE); // Bellek hatası durumunda programı sonlandır
			}

    	free(prob_estimates); // Belleği serbest bırak

        } else {

            predict_label = svm_predict(model, x);
            //fprintf(output, "%.17g\n", predict_label);
        }

        if (predict_label == -1.0 )
        
            printf("***************************  The individual is predicted to be a ADULT \a *************************** \n");
        else 
        
            printf("***************************  The individual is predicted to be a CHILD \a *************************** \n");

        for (int i = 0; i < max_nr_attr; i++) {
            x[i].index = -1;
            x[i].value = 0.0;
        }

         if ( ( getchar()) == 'q')
        {
                flag = 0; 
                printf("Loop exited \n");
        }
        else 
                flag = 1; 
    }
    svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	//free(new_line);
	return 0;
}






/* 
// (1) -->    gcc -o my_program deneme.c -lsvm 
// (2) --->  ./svm-scale -l -1 -u 1 -s train_dataset1 > train.scale
// (3) --->  ./svm-train -s 0 -c 5 -t 2 -g 0.5 -e 0.1 train.scale
// (4) --->  ./my_program train_dataset1 train_dataset1.model out

*/


















