using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class wallmats : MonoBehaviour
{

    public Material _bricksmat;
    public Material _highlighted;
    public Material trans;

    Renderer rend;

    public RockBuildScoreManager _score;

    // Start is called before the first frame update
    void Start()
    {
        rend = GetComponent<Renderer>();
        rend.enabled = true;
        rend.sharedMaterial = trans;
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void OnHoverEnter()
    {
        rend.sharedMaterial = _highlighted;
    }


    public void OnHoverExit()
    {
        rend.material = trans;
    }

    public void OnSelectEnter(SelectEnterEventArgs args)
    {
        rend.sharedMaterial = _bricksmat;
        _score.AddPoint();
        Destroy(this.GetComponent<XRGrabInteractable>(),0.05f);
        Destroy(this);
    }



}
