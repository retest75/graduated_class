/* Chap03.5-2 get token function */
#define MAX_STACK_SIZE 100

typedef enum
{
	lparan,	rparan,                       // '(',  ')'
	plus,	minus,	times,	divide,	mod,  // '+', '-', '*', '/'
	eos,                                  // '/\0'
	operand
} precedence;

precedence get_token(char *symbol, int *n)
{
	extern char expression[MAX_STACK_SIZE];
	
	*symbol = expression[(*n)++];
	switch (*symbol)
	{
		case '(':
			return lparan;
			break;
		case ')':
			return rparan;
			break;
		case '+':
			return plus;
			break;
		case '-':
			return minus;
			break;
		case '*':
			return times;
			break;
		case '/':
			return divide;
			break;
		case '%':
			return mod;
			break;
		case '\0':
			return eos;	
			break;
		default:
			return operand;
	}
}
