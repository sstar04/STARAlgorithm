/*
 * vtk_visual.h
 *
 * This file implement a VTK function that visualize a regular cubic grid and the list of node pass in the argument
 *  Created on: 04/mar/2015
 *      Author: Sabrina Stella
 */

#ifndef VTK_VISUAL__H_
#define VTK_VISUAL__H_

#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include <vtkCellArray.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include "vtkPointData.h"
#include <vtkCamera.h>

#include  <vtkStructuredPoints.h>
#include <vtkDataSetMapper.h>
#include "vtkIntArray.h"
#include <vtkLookupTable.h>

#include <vector>
#include "src/CellSearch.h"

using namespace std ;


void ShowGraph( vector<uint>& node, Params4Visual* P){

    bool check = false;
	int NPoint = P->NtotCell;
	int s = P->s;    //numero di suddivisioni per lato

	// create a structured grid
	vtkSmartPointer<vtkStructuredPoints> grid = vtkSmartPointer<vtkStructuredPoints>::New();

	grid->SetOrigin(-0.5,-0.5,-0.5);
	grid->SetDimensions(s+1,s+1,s+1);
	grid->SetSpacing(1,1,1);

	 // Create a mapper and actor
	 vtkSmartPointer<vtkDataSetMapper> grid_mapper = vtkSmartPointer<vtkDataSetMapper>::New();
	 grid_mapper->SetInputData(grid);

	 vtkSmartPointer<vtkActor> grid_actor = vtkSmartPointer<vtkActor>::New();
	 grid_actor->SetMapper(grid_mapper);
	 grid_actor->GetProperty()->SetOpacity(0.5);
	 grid_actor->GetProperty()->SetRepresentationToWireframe();


	//Draw a point at the center of each subdivision cube
	//define geometry (point)
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkIntArray> node_label = vtkSmartPointer<vtkIntArray>::New(); // assegno un valore ad un punto
	node_label->SetName("node_label");

	int hash;
	int hash_vector[NPoint];
	int index = 0;

	// Load the point, cell, and data attributes.
	for (int i=0; i<s; i++){
		for(int j=0; j<s; j++){
			for(int k=0; k<s; k++){
				points->InsertNextPoint(i,j,k);  //vector storing the point position

				//calculate hash value
				if(check){
					hash = (i *s *s) + (j * s) + k ;
					hash_vector[index] = hash;
					cout << "indice " << index << "ha un hash value= " << hash_vector[index] << endl;

				}
				node_label->InsertNextTuple1(node[index]);
				index++;
			}
		}
	}


	//create a dataset object (PolyData)
	vtkSmartPointer<vtkPolyData> graph = vtkSmartPointer<vtkPolyData>::New();
	graph->SetPoints(points);
	graph->GetPointData()->SetScalars(node_label);   // point attrbute data


	vtkSmartPointer<vtkVertexGlyphFilter> vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	vertexGlyphFilter->AddInputData(graph);

	// Create a lookup table to map cell data to colors
	vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
	int tableSize = 11; // 10 deafoul colour
	lut->SetNumberOfTableValues(tableSize);

	lut->Build();

	     // Fill in a few known colors, the rest will be generated if needed
	    lut->SetTableValue(0,   0,0,0, 0);  // transparent
	    lut->SetTableValue(1,  1,0,0, 1); // red
	    lut->SetTableValue(2,  0,0,0, 1); // green
	    lut->SetTableValue(3,  0,0,1, 1); // blue
	    lut->SetTableValue(4,  0,1,1, 1); // ciano
	    lut->SetTableValue(5,  1,1,0, 1); // yellow
	    lut->SetTableValue(6,  1,0,1, 1); // magenta
	    lut->SetTableValue(7,  0.3,0.3,0.3, 1); //
	    lut->SetTableValue(8,  1,0.1,0.1, 1); //


	// Create a mapper and actor
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());
	mapper->SetScalarRange(0, tableSize - 1);
	mapper->SetLookupTable(lut);
	//mapper->SetScalarModeToUsePointFieldData();
	mapper->ColorByArrayComponent("node_label",1);


	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetPointSize(4);


	///////////////
	/// RENDERING
	////////////////

	vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
		//camera->SetPosition(0, 0, 20);
		//camera->SetFocalPoint(0, 0, 0);
		//camera->SetViewAngle(45);
		//camera->SetViewUp(0.5,0.5, 0);

	// Create a renderer, render window, and interactor
	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	//renderer->SetActiveCamera(camera);
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	// Add the actor to the scene
	renderer->AddActor(actor);
	renderer->AddActor(grid_actor);
	//renderer->SetBackground(.3, .6, .3); // Background color green
	renderer->SetBackground(1, 1, 1); // Background color white

	// Render and interact
	renderWindow->Render();
	renderWindowInteractor->Start();

	return;

}


#endif /* VTK_VISUAL__H_ */
