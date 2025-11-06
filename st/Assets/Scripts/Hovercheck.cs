using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class Hovercheck : MonoBehaviour
{

    public bool _isHovering = false;
    [SerializeField] public XRBaseInteractable _interactableobject;


    public void OnHoverEntered(HoverEnterEventArgs args)
    {

        // Debug.Log($"{args.interactorObject} hovered over {args.interactableObject}", this);
        _isHovering = true;
        _interactableobject = (XRBaseInteractable)args.interactableObject;
    }

    public void OnHoverExited(HoverExitEventArgs args)
    {
        // Debug.Log($"{args.interactorObject} stopped hovering over {args.interactableObject}", this);
        _isHovering = false;
        _interactableobject = (XRBaseInteractable)args.interactableObject;
    }


}
