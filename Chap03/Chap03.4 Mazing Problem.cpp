/* Chap03.4 Mazing Problem */

#include <stdio.h>
#include <stdlib.h>
#define MAX_STACK_SIZE 100
#define MAZE_ROW 12
#define MAZE_COL 17
#define EXIT_ROW 12
#define EXIT_COL 17


/* maze amd mark */
int maze[MAZE_ROW+2][MAZE_COL+2] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
									{1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
				            		{1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1},
				            		{1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1},
				           	 		{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1},
				           	 		{1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
				            		{1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1},
				            		{1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
									{1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1},
									{1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1},
									{1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1},
									{1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1},
									{1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1},
									{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
/* path record */
int mark[MAZE_ROW+2][MAZE_COL+2] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
									{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
				            		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
				            		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
				           	 		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
				           	 		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
				            		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
				            		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
									{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
									{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
									{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
									{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
									{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
									{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};;

/* offest */
typedef struct
{
	short int vert;   // vertical move(row change)
	short int horiz;  // horizontal move(column change)
} offset;
/* move direction :  N        NE       E      ES      S        WS      W         WN */
offset move[8] = {{-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}};

/* define a stack implied route record(走過的路徑) */
typedef struct
{
	short int row; // current row
	short int col; // current col
	short int dir; // next direction(0 to 7)
} element;
element stack[MAX_STACK_SIZE];
int top = 0;

/* prototype */
element pop(int *);
void push(int *, element);
void path(void);



int main(void)
{
	path();
	system("pause");
	return 0;
}

void path(void)
{
	int i;
	int row, col;                 // current position
	int next_row, next_col, dir;  // next position and direction
	int found=0;                  // not found exit
	element position;             // path can pass(可以走的路徑)
	//int top=0;
	
	mark[1][1] = 1;   // row and column of entrance
	stack[0].row = 1; stack[0].col = 1; stack[0].dir = 1; // first go EN
	//printf("row col dir top\n");
	while (top>-1 && !found)
	{
		//printf("1,%2d%5d%5d%5d\n", stack[top].row, stack[top].col, stack[top].dir, top);
		position = pop(&top);
		//printf("2,%2d%5d%5d%5d\n", stack[top].row, stack[top].col, stack[top].dir, top);
		row = position.row; col = position.col; dir = position.dir; // current position and next direction
		//printf("%d %d %d \n", row, col, dir);
		while(dir<8 && !found) // if didn't find route after undergoing 8 direction, then left loop and try previous route(stack)
		{
			next_row = row + move[dir].vert;
			next_col  = col + move[dir].horiz;
			if (next_row == EXIT_ROW && next_col == EXIT_COL) // find exit
				found = 1;
			else if (!maze[next_row][next_col] && !mark[next_row][next_col]) // route can go and never go
			{
				mark[next_row][next_col] = 1; // already pass
				position.row = row; position.col = col; position.dir = ++dir; // stack current position and next direction
				push(&top, position);         // reset route record
				row = next_row; col = next_col; dir=0; // offset to next position and reset direction
				//printf("3,%2d%5d%5d%5d\n", stack[top].row, stack[top].col, stack[top].dir, top);
			}
			else
				++dir;
		}
	}
	/* find exit */
	if (found)
	{ 
	 	printf("The path is: ");
	 	printf("row col\n");
	 	for (i=0; i<=top; i++)
	 		printf("%15d%4d\n", stack[i].row, stack[i].col);
		printf("%15d%4d\n", row, col);
		printf("%15d%4d\n", EXIT_ROW, EXIT_COL);
	}
	else
		printf("The maze does not have a path\n");
	
}
element pop(int *top_ptr) // return pop element from the stack
{
	if (*top_ptr == -1)
	{
		printf("the stack is empty");
		element empty_stack = {0};
		return empty_stack;
	}
	else
		return stack[(*top_ptr)--];
}
void push(int *top_ptr, element item)
{
	if (*top_ptr == MAX_STACK_SIZE)
	{
		printf("th stack is full");
		return;
	}
	else
		stack[++*top_ptr] = item;
}


